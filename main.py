"""
Advanced Telegram bot "Golostickery" main module.
Provides comprehensive application lifecycle management with monitoring and error recovery.
"""
import os
# Fix OMP warning on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import asyncio
import signal
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import platform
import psutil
import aiohttp
from contextlib import asynccontextmanager
import json

from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.storage.redis import RedisStorage
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web
import redis.asyncio as redis

# Import configurations and modules
from config import config
from logger import setup_logging, get_logger, log_context
from db_manager.manager import DatabaseManager
from bot_handlers import register_handlers

# For model preloading
from stt_service.service import load_whisper_model, get_transcription_stats
from image_generation_service.service import (
    load_stable_diffusion_pipeline,
    get_generation_stats
)
# Removed import: clear_cache as clear_image_cache

# Setup logging first
setup_logging(
    structured_logs=config.environment.value == "production",
    async_file_handler=True
)
logger = get_logger(__name__)


class HealthChecker:
    """System health monitoring."""
    
    def __init__(self):
        """Initialize health checker."""
        self.checks: Dict[str, bool] = {
            'database': False,
            'whisper_model': False,
            'sd_model': False,
            'telegram_api': False,
            'disk_space': True,
            'memory': True
        }
        self.last_check = datetime.now()
        self.check_interval = 60  # seconds
    
    async def check_all(self) -> Dict[str, Any]:
        """Run all health checks."""
        logger.debug("Running health checks...")
        
        # System resources
        self.checks['disk_space'] = await self._check_disk_space()
        self.checks['memory'] = await self._check_memory()
        
        # Application components
        # These are set by the application during startup
        
        self.last_check = datetime.now()
        
        return {
            'status': 'healthy' if all(self.checks.values()) else 'unhealthy',
            'checks': self.checks.copy(),
            'last_check': self.last_check.isoformat(),
            'system': await self._get_system_info()
        }
    
    async def _check_disk_space(self) -> bool:
        """Check available disk space."""
        try:
            usage = psutil.disk_usage(str(config.paths.storage_dir))
            # Alert if less than 1GB free
            return usage.free > 1024 * 1024 * 1024
        except Exception as e:
            logger.warning(f"Disk space check failed: {e}")
            return False
    
    async def _check_memory(self) -> bool:
        """Check available memory."""
        try:
            memory = psutil.virtual_memory()
            # Alert if less than 10% available
            return memory.percent < 90
        except Exception as e:
            logger.warning(f"Memory check failed: {e}")
            return False
    
    async def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage(str(config.paths.storage_dir))
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_free_gb': disk.free / (1024**3),
                'disk_percent': disk.percent,
                'python_version': platform.python_version(),
                'platform': platform.platform()
            }
        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
            return {}
    
    def set_component_health(self, component: str, healthy: bool):
        """Set health status for a component."""
        self.checks[component] = healthy


class MetricsCollector:
    """Application metrics collection."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.metrics = {
            'start_time': datetime.now(),
            'requests_total': 0,
            'requests_success': 0,
            'requests_failed': 0,
            'active_users': set(),
            'generations_total': 0,
            'transcriptions_total': 0,
            'errors': []
        }
        self._lock = asyncio.Lock()
    
    async def increment(self, metric: str, value: int = 1):
        """Increment a metric."""
        async with self._lock:
            if metric in self.metrics and isinstance(self.metrics[metric], (int, float)):
                self.metrics[metric] += value
    
    async def add_user(self, user_id: int):
        """Add active user."""
        async with self._lock:
            self.metrics['active_users'].add(user_id)
    
    async def add_error(self, error: Dict[str, Any]):
        """Add error to metrics."""
        async with self._lock:
            self.metrics['errors'].append({
                **error,
                'timestamp': datetime.now().isoformat()
            })
            # Keep only last 100 errors
            if len(self.metrics['errors']) > 100:
                self.metrics['errors'] = self.metrics['errors'][-100:]
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        async with self._lock:
            uptime = (datetime.now() - self.metrics['start_time']).total_seconds()
            
            return {
                'uptime_seconds': uptime,
                'uptime_hours': uptime / 3600,
                'requests': {
                    'total': self.metrics['requests_total'],
                    'success': self.metrics['requests_success'],
                    'failed': self.metrics['requests_failed'],
                    'success_rate': (
                        self.metrics['requests_success'] / self.metrics['requests_total']
                        if self.metrics['requests_total'] > 0 else 0
                    )
                },
                'active_users_count': len(self.metrics['active_users']),
                'generations_total': self.metrics['generations_total'],
                'transcriptions_total': self.metrics['transcriptions_total'],
                'recent_errors': self.metrics['errors'][-10:]  # Last 10 errors
            }


class BotApplication:
    """Main application class with lifecycle management."""
    
    def __init__(self):
        """Initialize application."""
        self.bot: Optional[Bot] = None
        self.dp: Optional[Dispatcher] = None
        self.db_manager: Optional[DatabaseManager] = None
        self.health_checker = HealthChecker()
        self.metrics = MetricsCollector()
        self.web_app: Optional[web.Application] = None
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._tasks: List[asyncio.Task] = []
    
    async def setup(self):
        """Initialize all application components."""
        logger.info("=" * 80)
        logger.info("🚀 Starting Telegram Bot 'Golostickery'")
        logger.info(f"Environment: {config.environment.value}")
        logger.info(f"Python: {platform.python_version()}")
        logger.info(f"Platform: {platform.platform()}")
        logger.info("=" * 80)
        
        try:
            # 1. System checks
            await self._check_system_requirements()
            
            # 2. Initialize database
            await self._init_database()
            
            # 3. Initialize bot
            await self._init_bot()
            
            # 4. Load ML models
            await self._load_models()
            
            # 5. Setup web server if webhook mode
            if config.telegram.use_webhook:
                await self._setup_webhook()
            
            # 6. Register handlers
            self._register_handlers()
            
            # 7. Start background tasks
            await self._start_background_tasks()
            
            logger.info("✅ Application initialized successfully")
            
        except Exception as e:
            logger.critical(f"Failed to initialize application: {e}", exc_info=True)
            raise
    
    async def _check_system_requirements(self):
        """Check system requirements."""
        logger.info("Checking system requirements...")
        
        # Check Python version
        python_version = tuple(map(int, platform.python_version().split('.')[:2]))
        if python_version < (3, 10):
            raise RuntimeError(f"Python 3.10+ required, got {platform.python_version()}")
        
        # Check disk space
        disk_usage = psutil.disk_usage(str(config.paths.base_dir))
        free_gb = disk_usage.free / (1024**3)
        if free_gb < 2:
            logger.warning(f"Low disk space: {free_gb:.1f} GB free")
        
        # Check memory
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        if available_gb < 2:
            logger.warning(f"Low memory: {available_gb:.1f} GB available")
        
        # Check FFmpeg
        import shutil
        if not shutil.which("ffmpeg"):
            logger.warning("FFmpeg not found. Audio processing may not work.")
        
        # Check network connectivity
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://api.telegram.org", timeout=5) as resp:
                    if resp.status != 200:
                        logger.warning("Cannot reach Telegram API")
        except Exception as e:
            logger.warning(f"Network check failed: {e}")
        
        logger.info("✓ System requirements checked")
    
    async def _init_database(self):
        """Initialize database."""
        logger.info("Initializing database...")
        
        self.db_manager = DatabaseManager(config.paths.db_path)
        
        try:
            await self.db_manager.initialize()
            
            # Run maintenance
            # await self.db_manager.cleanup_old_data(days=30)  # Commented out - column 'resolved' issue
            
            # Get stats
            stats = await self.db_manager.get_statistics()
            logger.info(
                f"✓ Database initialized. "
                f"Users: {stats.get('total_users', 0)}, "
                f"Stickers: {stats.get('total_stickers', 0)}"
            )
            
            self.health_checker.set_component_health('database', True)
            
        except Exception as e:
            self.health_checker.set_component_health('database', False)
            raise RuntimeError(f"Database initialization failed: {e}")
    
    async def _init_bot(self):
        """Initialize bot and dispatcher."""
        logger.info("Initializing bot...")
        
        # Create bot without custom session to avoid timeout issues
        self.bot = Bot(
            token=config.telegram.bot_token,
            default=DefaultBotProperties(
                parse_mode=ParseMode.MARKDOWN,
                link_preview_is_disabled=True,
                allow_sending_without_reply=True
            )
        )
        
        # Create storage
        if config.environment.value == "production" and os.getenv('USE_REDIS', 'false').lower() == 'true':
            # Use Redis for production only if explicitly enabled
            try:
                redis_client = redis.Redis(
                    host=os.getenv('REDIS_HOST', 'localhost'),
                    port=int(os.getenv('REDIS_PORT', 6379)),
                    decode_responses=True
                )
                storage = RedisStorage(redis_client)
                logger.info("Using Redis storage")
            except Exception as e:
                logger.warning(f"Redis unavailable: {e}, using memory storage")
                storage = MemoryStorage()
        else:
            storage = MemoryStorage()
            logger.info("Using memory storage")
        
        # Create dispatcher
        self.dp = Dispatcher(storage=storage)
        
        # Add dependencies to context
        self.dp["bot"] = self.bot
        self.dp["db_manager"] = self.db_manager
        self.dp["health_checker"] = self.health_checker
        self.dp["metrics"] = self.metrics
        self.dp["storage_dir"] = str(config.paths.storage_dir)
        
        # Test bot connection
        try:
            bot_info = await self.bot.get_me()
            logger.info(
                f"✓ Bot initialized: @{bot_info.username} "
                f"(ID: {bot_info.id})"
            )
            self.health_checker.set_component_health('telegram_api', True)
        except Exception as e:
            self.health_checker.set_component_health('telegram_api', False)
            raise RuntimeError(f"Cannot connect to Telegram: {e}")
    
    async def _load_models(self):
        """Load ML models with progress tracking."""
        logger.info("Loading ML models...")
        
        # Load Whisper
        try:
            logger.info("• Loading Whisper model...")
            start_time = asyncio.get_event_loop().time()
            
            await asyncio.wait_for(
                load_whisper_model(),
                timeout=config.performance.whisper_timeout_sec * 2
            )
            
            load_time = asyncio.get_event_loop().time() - start_time
            logger.info(f"✓ Whisper loaded in {load_time:.1f}s")
            self.health_checker.set_component_health('whisper_model', True)
            
        except asyncio.TimeoutError:
            logger.error("Whisper loading timeout")
            self.health_checker.set_component_health('whisper_model', False)
            if config.environment.value == "production":
                raise
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            self.health_checker.set_component_health('whisper_model', False)
            if config.environment.value == "production":
                raise
        
        # Load Stable Diffusion
        try:
            logger.info("• Loading Stable Diffusion model...")
            start_time = asyncio.get_event_loop().time()
            
            await asyncio.wait_for(
                load_stable_diffusion_pipeline(),
                timeout=config.performance.sd_timeout_sec * 2
            )
            
            load_time = asyncio.get_event_loop().time() - start_time
            logger.info(f"✓ Stable Diffusion loaded in {load_time:.1f}s")
            self.health_checker.set_component_health('sd_model', True)
            
        except asyncio.TimeoutError:
            logger.error("Stable Diffusion loading timeout")
            self.health_checker.set_component_health('sd_model', False)
            if config.environment.value == "production":
                raise
        except Exception as e:
            logger.error(f"Failed to load Stable Diffusion: {e}")
            self.health_checker.set_component_health('sd_model', False)
            if config.environment.value == "production":
                raise
    
    async def _setup_webhook(self):
        """Setup webhook mode."""
        logger.info("Setting up webhook...")
        
        # Create web application
        self.web_app = web.Application()
        
        # Add routes
        self.web_app.router.add_get('/health', self._health_handler)
        self.web_app.router.add_get('/metrics', self._metrics_handler)
        
        # Setup webhook handler
        webhook_path = f"/webhook/{config.telegram.bot_token}"
        webhook_handler = SimpleRequestHandler(
            dispatcher=self.dp,
            bot=self.bot
        )
        webhook_handler.register(self.web_app, path=webhook_path)
        
        # Set webhook
        webhook_url = f"{config.telegram.webhook_url}{webhook_path}"
        await self.bot.set_webhook(
            url=webhook_url,
            drop_pending_updates=config.telegram.drop_pending_updates,
            allowed_updates=config.telegram.allowed_updates
        )
        
        logger.info(f"✓ Webhook set: {webhook_url}")
    
    def _register_handlers(self):
        """Register bot handlers."""
        logger.info("Registering handlers...")
        
        # Register all handlers
        register_handlers(self.dp)
        
        # Add middleware for metrics
        @self.dp.message.outer_middleware()
        async def metrics_middleware(handler, message, data):
            """Track metrics for all messages."""
            await self.metrics.increment('requests_total')
            await self.metrics.add_user(message.from_user.id)
            
            try:
                result = await handler(message, data)
                await self.metrics.increment('requests_success')
                return result
            except Exception as e:
                await self.metrics.increment('requests_failed')
                await self.metrics.add_error({
                    'type': type(e).__name__,
                    'message': str(e),
                    'user_id': message.from_user.id
                })
                raise
        
        # Add startup/shutdown handlers
        self.dp.startup.register(self._on_startup)
        self.dp.shutdown.register(self._on_shutdown)
        
        logger.info("✓ Handlers registered")
    
    async def _start_background_tasks(self):
        """Start background tasks."""
        logger.info("Starting background tasks...")
        
        # Periodic cleanup
        self._tasks.append(
            asyncio.create_task(self._periodic_cleanup())
        )
        
        # Health monitoring
        self._tasks.append(
            asyncio.create_task(self._health_monitor())
        )
        
        # Metrics reporting
        if config.environment.value == "production":
            self._tasks.append(
                asyncio.create_task(self._metrics_reporter())
            )
        
        logger.info(f"✓ Started {len(self._tasks)} background tasks")
    
    async def _periodic_cleanup(self):
        """Periodic cleanup task."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(config.performance.cleanup_interval_hours * 3600)
                
                logger.info("Running periodic cleanup...")
                
                # Clean old files
                from sticker_utils.utils import cleanup_old_files
                deleted_files = await cleanup_old_files(
                    str(config.paths.storage_dir),
                    days=config.performance.cleanup_keep_days
                )
                
                # Clean database
                await self.db_manager.cleanup_old_data(
                    days=config.performance.cleanup_keep_days
                )
                
                # Vacuum database
                await self.db_manager.vacuum()
                
                # Note: Image cache clearing removed - not implemented
                # If you need cache clearing, implement it in image_generation_service
                
                logger.info(f"Cleanup completed. Deleted {deleted_files} files")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    async def _health_monitor(self):
        """Health monitoring task."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Check every minute
                
                health_status = await self.health_checker.check_all()
                
                if health_status['status'] == 'unhealthy':
                    unhealthy_components = [
                        k for k, v in health_status['checks'].items() if not v
                    ]
                    logger.warning(f"Unhealthy components: {unhealthy_components}")
                    
                    # Try to recover
                    await self._try_recover_components(unhealthy_components)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
    
    async def _try_recover_components(self, components: List[str]):
        """Try to recover unhealthy components."""
        for component in components:
            try:
                if component == 'database':
                    # Try to reconnect
                    await self.db_manager.close()
                    await self.db_manager.initialize()
                    self.health_checker.set_component_health('database', True)
                    logger.info("Database connection recovered")
                    
                elif component == 'telegram_api':
                    # Test API connection
                    await self.bot.get_me()
                    self.health_checker.set_component_health('telegram_api', True)
                    logger.info("Telegram API connection recovered")
                    
            except Exception as e:
                logger.error(f"Failed to recover {component}: {e}")
    
    async def _metrics_reporter(self):
        """Report metrics to external service."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                metrics = await self.metrics.get_metrics()
                
                # Add service-specific metrics
                metrics['whisper_stats'] = await get_transcription_stats()
                metrics['sd_stats'] = await get_generation_stats()
                metrics['db_stats'] = await self.db_manager.get_statistics()
                
                # Log metrics (in production, send to monitoring service)
                logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics reporter error: {e}")
    
    async def _health_handler(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        health_status = await self.health_checker.check_all()
        
        status_code = 200 if health_status['status'] == 'healthy' else 503
        
        return web.json_response(health_status, status=status_code)
    
    async def _metrics_handler(self, request: web.Request) -> web.Response:
        """Metrics endpoint."""
        metrics = await self.metrics.get_metrics()
        
        # Add current health status
        metrics['health'] = await self.health_checker.check_all()
        
        return web.json_response(metrics)
    
    async def _on_startup(self):
        """Bot startup callback."""
        self._running = True
        logger.info("🚀 Bot started successfully!")
        
        # Send notification to admin
        if admin_id := os.getenv('ADMIN_USER_ID'):
            try:
                await self.bot.send_message(
                    admin_id,
                    f"🚀 Bot started!\n"
                    f"Environment: {config.environment.value}\n"
                    f"Models loaded: ✅"
                )
            except Exception as e:
                logger.warning(f"Failed to notify admin: {e}")
    
    async def _on_shutdown(self):
        """Bot shutdown callback."""
        self._running = False
        logger.info("Shutting down bot...")
        
        # Signal shutdown to background tasks
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
        
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Close connections
        if self.db_manager:
            await self.db_manager.close()
            logger.info("✓ Database closed")
        
        if self.bot:
            await self.bot.close()
            logger.info("✓ Bot session closed")
        
        logger.info("👋 Bot shutdown complete")
    
    async def run_polling(self):
        """Run bot in polling mode."""
        logger.info("Starting bot in polling mode...")
        
        await self.dp.start_polling(
            self.bot,
            allowed_updates=config.telegram.allowed_updates,
            drop_pending_updates=config.telegram.drop_pending_updates,
            handle_signals=False  # We handle signals ourselves
        )
    
    async def run_webhook(self):
        """Run bot in webhook mode."""
        logger.info("Starting bot in webhook mode...")
        
        runner = web.AppRunner(self.web_app)
        await runner.setup()
        
        site = web.TCPSite(
            runner,
            host='0.0.0.0',
            port=config.telegram.webhook_port
        )
        
        await site.start()
        logger.info(f"Webhook server started on port {config.telegram.webhook_port}")
        
        # Wait for shutdown
        await self._shutdown_event.wait()
        
        await runner.cleanup()
    
    async def run(self):
        """Main run method."""
        try:
            await self.setup()
            
            if config.telegram.use_webhook:
                await self.run_webhook()
            else:
                await self.run_polling()
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.critical(f"Critical error: {e}", exc_info=True)
            raise
        finally:
            if self._running:
                await self._on_shutdown()


def setup_signal_handlers(app: BotApplication):
    """Setup system signal handlers."""
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}")
        asyncio.create_task(app._on_shutdown())
        sys.exit(0)
    
    # Register handlers
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, signal_handler)
    
    # Windows specific
    if sys.platform == "win32":
        signal.signal(signal.SIGBREAK, signal_handler)


@asynccontextmanager
async def lifespan(app: BotApplication):
    """Application lifespan context manager."""
    # Startup
    yield
    # Shutdown is handled by the app


async def main():
    """Main entry point."""
    app = BotApplication()
    
    # Setup signal handlers
    setup_signal_handlers(app)
    
    try:
        await app.run()
    except Exception as e:
        logger.critical(f"Application failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # Platform-specific optimizations
    if sys.platform == "win32":
        # Windows ProactorEventLoop for better subprocess support
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    else:
        # Enable uvloop on Unix for better performance
        try:
            import uvloop
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            logger.info("Using uvloop for better performance")
        except ImportError:
            pass
    
    # Run application
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)