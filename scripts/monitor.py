#!/usr/bin/env python3
"""
SignaMentis Real-time Monitoring Script
Provides live monitoring of services, resources, and system health
"""

import argparse
import subprocess
import sys
import time
import json
import psutil
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import docker
from docker.errors import DockerException
import threading
import queue
import curses
from collections import deque


class SystemMonitor:
    """Real-time system monitoring for SignaMentis"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.client = None
        self.monitoring = False
        self.data_queue = queue.Queue()
        self.metrics_history = deque(maxlen=1000)
        
        try:
            self.client = docker.from_env()
        except DockerException as e:
            print(f"Warning: Docker client not available: {e}")
    
    def log(self, message: str, level: str = "INFO") -> None:
        """Log monitoring messages"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
    
    def get_system_metrics(self) -> Dict:
        """Get current system metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            # Disk usage
            disk = psutil.disk_usage('.')
            disk_percent = disk.percent
            disk_used_gb = disk.used / (1024**3)
            disk_total_gb = disk.total / (1024**3)
            
            # Network I/O
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv
            
            return {
                "timestamp": datetime.now().isoformat(),
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count
                },
                "memory": {
                    "percent": memory_percent,
                    "used_gb": memory_used_gb,
                    "total_gb": memory_total_gb
                },
                "disk": {
                    "percent": disk_percent,
                    "used_gb": disk_used_gb,
                    "total_gb": disk_total_gb
                },
                "network": {
                    "bytes_sent": network_bytes_sent,
                    "bytes_recv": network_bytes_recv
                }
            }
        except Exception as e:
            self.log(f"❌ Failed to get system metrics: {e}", "ERROR")
            return {}
    
    def get_docker_metrics(self) -> Dict:
        """Get Docker container metrics"""
        if not self.client:
            return {}
        
        try:
            containers = self.client.containers.list()
            container_metrics = {}
            
            for container in containers:
                try:
                    stats = container.stats(stream=False)
                    
                    # CPU usage
                    cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                               stats['precpu_stats']['cpu_usage']['total_usage']
                    system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                                 stats['precpu_stats']['system_cpu_usage']
                    
                    cpu_percent = 0
                    if system_delta > 0:
                        cpu_percent = (cpu_delta / system_delta) * 100
                    
                    # Memory usage
                    memory_usage = stats['memory_stats']['usage'] / (1024 * 1024)  # MB
                    memory_limit = stats['memory_stats']['limit'] / (1024 * 1024) if stats['memory_stats'].get('limit') else 0
                    memory_percent = (memory_usage / memory_limit * 100) if memory_limit > 0 else 0
                    
                    # Network I/O
                    network_rx = stats['networks'].get('eth0', {}).get('rx_bytes', 0) / (1024 * 1024)  # MB
                    network_tx = stats['networks'].get('eth0', {}).get('tx_bytes', 0) / (1024 * 1024)  # MB
                    
                    container_metrics[container.name] = {
                        "status": container.status,
                        "cpu_percent": cpu_percent,
                        "memory_mb": memory_usage,
                        "memory_percent": memory_percent,
                        "network_rx_mb": network_rx,
                        "network_tx_mb": network_tx,
                        "created": container.attrs['Created']
                    }
                    
                except Exception as e:
                    self.log(f"❌ Failed to get metrics for {container.name}: {e}", "ERROR")
            
            return container_metrics
            
        except Exception as e:
            self.log(f"❌ Failed to get Docker metrics: {e}", "ERROR")
            return {}
    
    def get_service_health(self) -> Dict:
        """Get service health status"""
        services = {
            "trading_system": "http://localhost:8000/health",
            "news_nlp": "http://localhost:8001/health",
            "mlflow": "http://localhost:5000/health",
            "grafana": "http://localhost:3000/api/health",
            "prometheus": "http://localhost:9090/-/healthy"
        }
        
        health_status = {}
        
        for service_name, url in services.items():
            try:
                response = requests.get(url, timeout=5)
                health_status[service_name] = {
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "response_time": response.elapsed.total_seconds() * 1000,  # ms
                    "status_code": response.status_code,
                    "last_check": datetime.now().isoformat()
                }
            except Exception as e:
                health_status[service_name] = {
                    "status": "unreachable",
                    "error": str(e),
                    "last_check": datetime.now().isoformat()
                }
        
        return health_status
    
    def get_database_metrics(self) -> Dict:
        """Get database performance metrics"""
        metrics = {}
        
        try:
            # MongoDB metrics
            mongo_result = subprocess.run([
                "docker", "compose", "exec", "-T", "mongodb", 
                "mongosh", "--eval", "db.stats()"
            ], capture_output=True, text=True, timeout=10)
            
            if mongo_result.returncode == 0:
                # Parse MongoDB stats
                output = mongo_result.stdout
                if "collections" in output:
                    metrics["mongodb"] = {
                        "status": "connected",
                        "collections": "available",
                        "last_check": datetime.now().isoformat()
                    }
                else:
                    metrics["mongodb"] = {"status": "error"}
            else:
                metrics["mongodb"] = {"status": "unreachable"}
                
        except Exception as e:
            metrics["mongodb"] = {"status": "error", "error": str(e)}
        
        try:
            # Redis metrics
            redis_result = subprocess.run([
                "docker", "compose", "exec", "-T", "redis", 
                "redis-cli", "info", "memory"
            ], capture_output=True, text=True, timeout=10)
            
            if redis_result.returncode == 0:
                # Parse Redis info
                output = redis_result.stdout
                used_memory = "unknown"
                for line in output.split('\n'):
                    if line.startswith('used_memory:'):
                        used_memory = int(line.split(':')[1]) / (1024 * 1024)  # MB
                        break
                
                metrics["redis"] = {
                    "status": "connected",
                    "used_memory_mb": used_memory,
                    "last_check": datetime.now().isoformat()
                }
            else:
                metrics["redis"] = {"status": "unreachable"}
                
        except Exception as e:
            metrics["redis"] = {"status": "error", "error": str(e)}
        
        return metrics
    
    def get_mlflow_metrics(self) -> Dict:
        """Get MLflow experiment and model metrics"""
        try:
            # Get experiments count
            experiments_response = requests.get(
                "http://localhost:5000/api/2.0/mlflow/experiments/list",
                timeout=10
            )
            
            if experiments_response.status_code == 200:
                experiments_data = experiments_response.json()
                experiments_count = len(experiments_data.get('experiments', []))
                
                # Get models count
                models_response = requests.get(
                    "http://localhost:5000/api/2.0/mlflow/registered-models/list",
                    timeout=10
                )
                
                models_count = 0
                if models_response.status_code == 200:
                    models_data = models_response.json()
                    models_count = len(models_data.get('registered_models', []))
                
                return {
                    "status": "connected",
                    "experiments_count": experiments_count,
                    "models_count": models_count,
                    "last_check": datetime.now().isoformat()
                }
            else:
                return {"status": "error", "status_code": experiments_response.status_code}
                
        except Exception as e:
            return {"status": "unreachable", "error": str(e)}
    
    def collect_metrics(self) -> Dict:
        """Collect all metrics"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "system": self.get_system_metrics(),
            "docker": self.get_docker_metrics(),
            "services": self.get_service_health(),
            "databases": self.get_database_metrics(),
            "mlflow": self.get_mlflow_metrics()
        }
        
        # Store in history
        self.metrics_history.append(metrics)
        
        return metrics
    
    def start_monitoring(self, interval: int = 5) -> None:
        """Start continuous monitoring"""
        self.monitoring = True
        self.log(f"Starting monitoring with {interval}s interval...")
        
        try:
            while self.monitoring:
                metrics = self.collect_metrics()
                self.data_queue.put(metrics)
                time.sleep(interval)
                
        except KeyboardInterrupt:
            self.log("Monitoring stopped by user")
        except Exception as e:
            self.log(f"❌ Monitoring error: {e}", "ERROR")
        finally:
            self.monitoring = False
    
    def stop_monitoring(self) -> None:
        """Stop monitoring"""
        self.monitoring = False
        self.log("Monitoring stopped")
    
    def get_metrics_summary(self) -> Dict:
        """Get summary of collected metrics"""
        if not self.metrics_history:
            return {}
        
        # Calculate averages over last 10 data points
        recent_metrics = list(self.metrics_history)[-10:]
        
        summary = {
            "total_samples": len(self.metrics_history),
            "recent_samples": len(recent_metrics),
            "monitoring_duration": "N/A"
        }
        
        if len(self.metrics_history) > 1:
            first_time = datetime.fromisoformat(self.metrics_history[0]["timestamp"])
            last_time = datetime.fromisoformat(self.metrics_history[-1]["timestamp"])
            duration = last_time - first_time
            summary["monitoring_duration"] = str(duration)
        
        # System averages
        if recent_metrics:
            cpu_values = [m["system"]["cpu"]["percent"] for m in recent_metrics if "system" in m]
            memory_values = [m["system"]["memory"]["percent"] for m in recent_metrics if "system" in m]
            
            if cpu_values:
                summary["avg_cpu_percent"] = sum(cpu_values) / len(cpu_values)
            if memory_values:
                summary["avg_memory_percent"] = sum(memory_values) / len(memory_values)
        
        return summary


class ConsoleMonitor:
    """Console-based monitoring interface"""
    
    def __init__(self, monitor: SystemMonitor):
        self.monitor = monitor
        self.screen = None
    
    def setup_curses(self):
        """Setup curses for console display"""
        self.screen = curses.initscr()
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)    # Healthy
        curses.init_pair(2, curses.COLOR_RED, -1)      # Unhealthy
        curses.init_pair(3, curses.COLOR_YELLOW, -1)   # Warning
        curses.init_pair(4, curses.COLOR_BLUE, -1)     # Info
        curses.init_pair(5, curses.COLOR_CYAN, -1)     # Metrics
        
        curses.noecho()
        curses.cbreak()
        curses.curs_set(0)
    
    def cleanup_curses(self):
        """Cleanup curses"""
        if self.screen:
            curses.nocbreak()
            curses.echo()
            curses.endwin()
    
    def draw_header(self, y: int) -> int:
        """Draw header information"""
        if not self.screen:
            return y
        
        self.screen.addstr(y, 0, "🚀 SignaMentis Real-time Monitor", curses.color_pair(4) | curses.A_BOLD)
        y += 1
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.screen.addstr(y, 0, f"📅 {timestamp}", curses.color_pair(4))
        y += 2
        
        return y
    
    def draw_system_metrics(self, y: int, metrics: Dict) -> int:
        """Draw system metrics"""
        if not self.screen or "system" not in metrics:
            return y
        
        self.screen.addstr(y, 0, "💻 System Metrics", curses.color_pair(5) | curses.A_BOLD)
        y += 1
        
        system = metrics["system"]
        if "cpu" in system:
            cpu_color = curses.color_pair(1) if system["cpu"]["percent"] < 80 else curses.color_pair(3)
            self.screen.addstr(y, 2, f"CPU: {system['cpu']['percent']:.1f}% ({system['cpu']['count']} cores)", cpu_color)
            y += 1
        
        if "memory" in system:
            mem_color = curses.color_pair(1) if system["memory"]["percent"] < 80 else curses.color_pair(3)
            self.screen.addstr(y, 2, f"Memory: {system['memory']['percent']:.1f}% ({system['memory']['used_gb']:.1f}GB / {system['memory']['total_gb']:.1f}GB)", mem_color)
            y += 1
        
        if "disk" in system:
            disk_color = curses.color_pair(1) if system["disk"]["percent"] < 80 else curses.color_pair(3)
            self.screen.addstr(y, 2, f"Disk: {system['disk']['percent']:.1f}% ({system['disk']['used_gb']:.1f}GB / {system['disk']['total_gb']:.1f}GB)", disk_color)
            y += 1
        
        y += 1
        return y
    
    def draw_service_health(self, y: int, metrics: Dict) -> int:
        """Draw service health status"""
        if not self.screen or "services" not in metrics:
            return y
        
        self.screen.addstr(y, 0, "🔍 Service Health", curses.color_pair(5) | curses.A_BOLD)
        y += 1
        
        services = metrics["services"]
        for service_name, status in services.items():
            if status["status"] == "healthy":
                color = curses.color_pair(1)
                status_icon = "✅"
            elif status["status"] == "unhealthy":
                color = curses.color_pair(2)
                status_icon = "❌"
            else:
                color = curses.color_pair(3)
                status_icon = "⚠️"
            
            response_time = status.get("response_time", "N/A")
            if isinstance(response_time, (int, float)):
                response_time = f"{response_time:.1f}ms"
            
            self.screen.addstr(y, 2, f"{status_icon} {service_name}: {status['status']} ({response_time})", color)
            y += 1
        
        y += 1
        return y
    
    def draw_docker_metrics(self, y: int, metrics: Dict) -> int:
        """Draw Docker container metrics"""
        if not self.screen or "docker" not in metrics:
            return y
        
        self.screen.addstr(y, 0, "🐳 Docker Containers", curses.color_pair(5) | curses.A_BOLD)
        y += 1
        
        containers = metrics["docker"]
        for container_name, container_metrics in containers.items():
            # Truncate long names
            display_name = container_name[:30] + "..." if len(container_name) > 30 else container_name
            
            status_color = curses.color_pair(1) if container_metrics["status"] == "running" else curses.color_pair(3)
            self.screen.addstr(y, 2, f"📦 {display_name}", status_color)
            y += 1
            
            # Container details
            cpu_percent = container_metrics.get("cpu_percent", 0)
            memory_mb = container_metrics.get("memory_mb", 0)
            
            self.screen.addstr(y, 4, f"CPU: {cpu_percent:.1f}% | Memory: {memory_mb:.1f}MB", curses.color_pair(4))
            y += 1
        
        y += 1
        return y
    
    def draw_summary(self, y: int) -> int:
        """Draw monitoring summary"""
        if not self.screen:
            return y
        
        summary = self.monitor.get_metrics_summary()
        
        self.screen.addstr(y, 0, "📊 Monitoring Summary", curses.color_pair(5) | curses.A_BOLD)
        y += 1
        
        if summary:
            self.screen.addstr(y, 2, f"Samples: {summary.get('total_samples', 0)}", curses.color_pair(4))
            y += 1
            
            if "avg_cpu_percent" in summary:
                self.screen.addstr(y, 2, f"Avg CPU: {summary['avg_cpu_percent']:.1f}%", curses.color_pair(4))
                y += 1
            
            if "avg_memory_percent" in summary:
                self.screen.addstr(y, 2, f"Avg Memory: {summary['avg_memory_percent']:.1f}%", curses.color_pair(4))
                y += 1
        
        y += 1
        return y
    
    def draw_controls(self, y: int) -> int:
        """Draw control instructions"""
        if not self.screen:
            return y
        
        self.screen.addstr(y, 0, "🎮 Controls", curses.color_pair(4) | curses.A_BOLD)
        y += 1
        self.screen.addstr(y, 2, "q: Quit | r: Refresh | s: Save metrics", curses.color_pair(4))
        y += 2
        
        return y
    
    def run_console(self, refresh_interval: int = 2):
        """Run console monitoring interface"""
        try:
            self.setup_curses()
            
            # Start monitoring in background
            monitor_thread = threading.Thread(
                target=self.monitor.start_monitoring,
                args=(refresh_interval,),
                daemon=True
            )
            monitor_thread.start()
            
            while True:
                # Clear screen
                self.screen.clear()
                
                # Get latest metrics
                try:
                    metrics = self.monitor.data_queue.get_nowait()
                except queue.Empty:
                    metrics = {}
                
                # Draw interface
                y = 0
                y = self.draw_header(y)
                y = self.draw_system_metrics(y, metrics)
                y = self.draw_service_health(y, metrics)
                y = self.draw_docker_metrics(y, metrics)
                y = self.draw_summary(y)
                y = self.draw_controls(y)
                
                # Refresh screen
                self.screen.refresh()
                
                # Handle input
                self.screen.timeout(1000)  # 1 second timeout
                try:
                    key = self.screen.getch()
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        # Force refresh
                        pass
                    elif key == ord('s'):
                        # Save metrics
                        self.save_metrics(metrics)
                except curses.error:
                    pass
                
        except KeyboardInterrupt:
            pass
        finally:
            self.monitor.stop_monitoring()
            self.cleanup_curses()
    
    def save_metrics(self, metrics: Dict):
        """Save current metrics to file"""
        if not metrics:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_file = self.monitor.project_root / "logs" / f"metrics_{timestamp}.json"
            metrics_file.parent.mkdir(exist_ok=True)
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Show save confirmation
            if self.screen:
                self.screen.addstr(self.screen.getmaxyx()[0] - 1, 0, f"💾 Metrics saved to {metrics_file.name}", curses.color_pair(1))
                self.screen.refresh()
                time.sleep(2)
                
        except Exception as e:
            if self.screen:
                self.screen.addstr(self.screen.getmaxyx()[0] - 1, 0, f"❌ Failed to save metrics: {e}", curses.color_pair(2))
                self.screen.refresh()
                time.sleep(2)


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="SignaMentis Real-time Monitor")
    parser.add_argument("--console", "-c", action="store_true", help="Run in console mode")
    parser.add_argument("--interval", "-i", type=int, default=5, help="Monitoring interval in seconds")
    parser.add_argument("--save", "-s", action="store_true", help="Save metrics to file")
    parser.add_argument("--summary", action="store_true", help="Show metrics summary")
    
    args = parser.parse_args()
    
    monitor = SystemMonitor()
    
    try:
        if args.console:
            # Console mode
            console_monitor = ConsoleMonitor(monitor)
            console_monitor.run_console(args.interval)
        
        elif args.summary:
            # Show summary
            summary = monitor.get_metrics_summary()
            print("📊 Monitoring Summary:")
            print("=" * 40)
            for key, value in summary.items():
                print(f"{key}: {value}")
        
        elif args.save:
            # Collect and save metrics
            metrics = monitor.collect_metrics()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_file = monitor.project_root / "logs" / f"metrics_{timestamp}.json"
            metrics_file.parent.mkdir(exist_ok=True)
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print(f"✅ Metrics saved to {metrics_file}")
        
        else:
            # Simple monitoring mode
            print("🚀 Starting SignaMentis monitoring...")
            print("Press Ctrl+C to stop")
            print()
            
            try:
                while True:
                    metrics = monitor.collect_metrics()
                    
                    # Display current metrics
                    print(f"\r📊 {datetime.now().strftime('%H:%M:%S')} | ", end="")
                    
                    if "system" in metrics:
                        cpu = metrics["system"].get("cpu", {}).get("percent", 0)
                        memory = metrics["system"].get("memory", {}).get("percent", 0)
                        print(f"CPU: {cpu:.1f}% | Memory: {memory:.1f}%", end="")
                    
                    if "services" in metrics:
                        healthy_services = sum(1 for s in metrics["services"].values() if s.get("status") == "healthy")
                        total_services = len(metrics["services"])
                        print(f" | Services: {healthy_services}/{total_services}", end="")
                    
                    print(" " * 20, end="\r")  # Clear line
                    
                    time.sleep(args.interval)
                    
            except KeyboardInterrupt:
                print("\n\nMonitoring stopped")
    
    except Exception as e:
        print(f"❌ Monitoring failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
