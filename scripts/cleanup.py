#!/usr/bin/env python3
"""
SignaMentis Cleanup and Maintenance Script
Manages Docker resources, cleanup, and system maintenance
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional
import docker
from docker.errors import DockerException


class CleanupManager:
    """Manages cleanup and maintenance of SignaMentis Docker resources"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.client = None
        
        try:
            self.client = docker.from_env()
        except DockerException as e:
            print(f"Warning: Docker client not available: {e}")
    
    def log(self, message: str, level: str = "INFO") -> None:
        """Log cleanup messages"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
    
    def run_command(self, command: List[str], check: bool = False) -> subprocess.CompletedProcess:
        """Run a shell command"""
        try:
            result = subprocess.run(
                command,
                check=check,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            return result
        except subprocess.CalledProcessError as e:
            self.log(f"Command failed: {' '.join(command)}", "ERROR")
            return e
    
    def cleanup_containers(self, force: bool = False) -> None:
        """Clean up stopped and unused containers"""
        self.log("Cleaning up containers...")
        
        try:
            # Stop all running containers
            self.log("Stopping running containers...")
            self.run_command(["docker", "compose", "down"], check=False)
            
            # Remove stopped containers
            if force:
                self.log("Force removing all containers...")
                self.run_command(["docker", "container", "prune", "-f"], check=False)
            else:
                self.log("Removing stopped containers...")
                self.run_command(["docker", "container", "prune"], check=False)
            
            self.log("✅ Container cleanup completed!")
            
        except Exception as e:
            self.log(f"❌ Container cleanup failed: {e}", "ERROR")
    
    def cleanup_images(self, force: bool = False, all_images: bool = False) -> None:
        """Clean up unused Docker images"""
        self.log("Cleaning up images...")
        
        try:
            if all_images:
                self.log("Removing all unused images...")
                self.run_command(["docker", "image", "prune", "-a", "-f" if force else ""], check=False)
            else:
                self.log("Removing dangling images...")
                self.run_command(["docker", "image", "prune", "-f" if force else ""], check=False)
            
            self.log("✅ Image cleanup completed!")
            
        except Exception as e:
            self.log(f"❌ Image cleanup failed: {e}", "ERROR")
    
    def cleanup_volumes(self, force: bool = False) -> None:
        """Clean up unused volumes"""
        self.log("Cleaning up volumes...")
        
        try:
            if force:
                self.log("Force removing all volumes...")
                self.run_command(["docker", "volume", "prune", "-f"], check=False)
            else:
                self.log("Removing unused volumes...")
                self.run_command(["docker", "volume", "prune"], check=False)
            
            self.log("✅ Volume cleanup completed!")
            
        except Exception as e:
            self.log(f"❌ Volume cleanup failed: {e}", "ERROR")
    
    def cleanup_networks(self, force: bool = False) -> None:
        """Clean up unused networks"""
        self.log("Cleaning up networks...")
        
        try:
            if force:
                self.log("Force removing all networks...")
                self.run_command(["docker", "network", "prune", "-f"], check=False)
            else:
                self.log("Removing unused networks...")
                self.run_command(["docker", "network", "prune"], check=False)
            
            self.log("✅ Network cleanup completed!")
            
        except Exception as e:
            self.log(f"❌ Network cleanup failed: {e}", "ERROR")
    
    def cleanup_build_cache(self) -> None:
        """Clean up Docker build cache"""
        self.log("Cleaning up build cache...")
        
        try:
            self.run_command(["docker", "builder", "prune", "-f"], check=False)
            self.log("✅ Build cache cleanup completed!")
            
        except Exception as e:
            self.log(f"❌ Build cache cleanup failed: {e}", "ERROR")
    
    def cleanup_logs(self, older_than_days: int = 7) -> None:
        """Clean up old log files"""
        self.log(f"Cleaning up logs older than {older_than_days} days...")
        
        try:
            # Clean up application logs
            logs_dir = self.project_root / "logs"
            if logs_dir.exists():
                import shutil
                from datetime import datetime, timedelta
                
                cutoff_date = datetime.now() - timedelta(days=older_than_days)
                removed_count = 0
                
                for log_file in logs_dir.glob("*.log"):
                    if log_file.stat().st_mtime < cutoff_date.timestamp():
                        log_file.unlink()
                        removed_count += 1
                
                self.log(f"✅ Removed {removed_count} old log files")
            
            # Clean up Docker logs
            self.run_command(["docker", "system", "prune", "-f"], check=False)
            self.log("✅ Docker logs cleanup completed!")
            
        except Exception as e:
            self.log(f"❌ Logs cleanup failed: {e}", "ERROR")
    
    def cleanup_data_directories(self, force: bool = False) -> None:
        """Clean up data directories"""
        self.log("Cleaning up data directories...")
        
        try:
            data_dirs = ["data", "cache", "models", "backups"]
            
            for dir_name in data_dirs:
                dir_path = self.project_root / dir_name
                if dir_path.exists():
                    if force:
                        import shutil
                        shutil.rmtree(dir_path)
                        self.log(f"✅ Removed {dir_name} directory")
                    else:
                        self.log(f"⚠️  {dir_name} directory exists (use --force to remove)")
            
            self.log("✅ Data directories cleanup completed!")
            
        except Exception as e:
            self.log(f"❌ Data directories cleanup failed: {e}", "ERROR")
    
    def cleanup_temporary_files(self) -> None:
        """Clean up temporary files"""
        self.log("Cleaning up temporary files...")
        
        try:
            # Clean up Python cache
            import shutil
            
            # Remove __pycache__ directories
            for pycache_dir in self.project_root.rglob("__pycache__"):
                shutil.rmtree(pycache_dir)
                self.log(f"✅ Removed {pycache_dir}")
            
            # Remove .pyc files
            for pyc_file in self.project_root.rglob("*.pyc"):
                pyc_file.unlink()
                self.log(f"✅ Removed {pyc_file}")
            
            # Remove .pyo files
            for pyo_file in self.project_root.rglob("*.pyo"):
                pyo_file.unlink()
                self.log(f"✅ Removed {pyo_file}")
            
            self.log("✅ Temporary files cleanup completed!")
            
        except Exception as e:
            self.log(f"❌ Temporary files cleanup failed: {e}", "ERROR")
    
    def cleanup_docker_system(self, force: bool = False) -> None:
        """Complete Docker system cleanup"""
        self.log("Performing complete Docker system cleanup...")
        
        try:
            # Stop all services
            self.log("Stopping all services...")
            self.run_command(["docker", "compose", "down"], check=False)
            
            # Remove all unused resources
            if force:
                self.log("Force removing all unused resources...")
                self.run_command(["docker", "system", "prune", "-a", "-f", "--volumes"], check=False)
            else:
                self.log("Removing unused resources...")
                self.run_command(["docker", "system", "prune", "-f"], check=False)
            
            self.log("✅ Docker system cleanup completed!")
            
        except Exception as e:
            self.log(f"❌ Docker system cleanup failed: {e}", "ERROR")
    
    def reset_environment(self, force: bool = False) -> None:
        """Reset the entire environment to clean state"""
        if not force:
            response = input("⚠️  This will remove ALL data and containers. Are you sure? (yes/no): ")
            if response.lower() != "yes":
                self.log("Reset cancelled by user", "WARNING")
                return
        
        self.log("🚨 Resetting entire environment...")
        
        try:
            # Stop and remove everything
            self.run_command(["docker", "compose", "down", "--volumes", "--remove-orphans"], check=False)
            
            # Remove all containers, images, volumes, and networks
            self.run_command(["docker", "system", "prune", "-a", "-f", "--volumes"], check=False)
            
            # Clean up data directories
            self.cleanup_data_directories(force=True)
            
            # Clean up temporary files
            self.cleanup_temporary_files()
            
            self.log("✅ Environment reset completed!")
            self.log("💡 Run 'python scripts/deploy.py deploy' to redeploy", "INFO")
            
        except Exception as e:
            self.log(f"❌ Environment reset failed: {e}", "ERROR")
    
    def show_system_info(self) -> None:
        """Show system information and resource usage"""
        self.log("📊 System Information:")
        self.log("=" * 40)
        
        try:
            # Docker info
            info_result = self.run_command(["docker", "info"], check=False)
            if info_result.returncode == 0:
                self.log("✅ Docker is running")
                
                # Parse Docker info
                lines = info_result.stdout.split('\n')
                for line in lines:
                    if 'Containers:' in line or 'Images:' in line or 'Volumes:' in line:
                        self.log(f"  {line.strip()}")
            else:
                self.log("❌ Docker is not running", "ERROR")
            
            # Disk usage
            disk_result = self.run_command(["df", "-h", "."], check=False)
            if disk_result.returncode == 0:
                self.log("\n💾 Disk Usage:")
                lines = disk_result.stdout.split('\n')
                for line in lines[1:]:  # Skip header
                    if line.strip():
                        self.log(f"  {line.strip()}")
            
            # Memory usage
            if self.client:
                try:
                    containers = self.client.containers.list()
                    total_memory = 0
                    
                    self.log("\n🐳 Container Memory Usage:")
                    for container in containers:
                        stats = container.stats(stream=False)
                        memory_usage = stats['memory_stats']['usage'] / (1024 * 1024)  # MB
                        total_memory += memory_usage
                        self.log(f"  {container.name}: {memory_usage:.1f} MB")
                    
                    self.log(f"  Total: {total_memory:.1f} MB")
                    
                except Exception as e:
                    self.log(f"❌ Could not get container stats: {e}", "ERROR")
            
        except Exception as e:
            self.log(f"❌ Could not get system info: {e}", "ERROR")
    
    def optimize_system(self) -> None:
        """Optimize system performance"""
        self.log("🔧 Optimizing system performance...")
        
        try:
            # Clean up build cache
            self.cleanup_build_cache()
            
            # Clean up unused resources
            self.cleanup_docker_system(force=False)
            
            # Clean up old logs
            self.cleanup_logs(older_than_days=3)
            
            # Clean up temporary files
            self.cleanup_temporary_files()
            
            self.log("✅ System optimization completed!")
            
        except Exception as e:
            self.log(f"❌ System optimization failed: {e}", "ERROR")
    
    def run_cleanup(self, cleanup_type: str, force: bool = False, **kwargs) -> None:
        """Run specific cleanup operation"""
        cleanup_functions = {
            "containers": lambda: self.cleanup_containers(force),
            "images": lambda: self.cleanup_images(force, kwargs.get("all_images", False)),
            "volumes": lambda: self.cleanup_volumes(force),
            "networks": lambda: self.cleanup_networks(force),
            "build-cache": lambda: self.cleanup_build_cache(),
            "logs": lambda: self.cleanup_logs(kwargs.get("older_than_days", 7)),
            "data": lambda: self.cleanup_data_directories(force),
            "temp": lambda: self.cleanup_temporary_files(),
            "system": lambda: self.cleanup_docker_system(force),
            "all": lambda: self.cleanup_docker_system(force),
            "reset": lambda: self.reset_environment(force),
            "optimize": lambda: self.optimize_system()
        }
        
        if cleanup_type in cleanup_functions:
            cleanup_functions[cleanup_type]()
        else:
            self.log(f"❌ Unknown cleanup type: {cleanup_type}", "ERROR")


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="SignaMentis Cleanup and Maintenance")
    parser.add_argument("action", choices=[
        "containers", "images", "volumes", "networks", "build-cache",
        "logs", "data", "temp", "system", "all", "reset", "optimize", "info"
    ], help="Cleanup action to perform")
    parser.add_argument("--force", "-f", action="store_true", help="Force cleanup without confirmation")
    parser.add_argument("--all-images", action="store_true", help="Remove all unused images (for images action)")
    parser.add_argument("--older-than-days", type=int, default=7, help="Remove logs older than N days (for logs action)")
    
    args = parser.parse_args()
    
    cleanup_manager = CleanupManager()
    
    try:
        if args.action == "info":
            cleanup_manager.show_system_info()
        else:
            cleanup_manager.run_cleanup(
                args.action,
                force=args.force,
                all_images=args.all_images,
                older_than_days=args.older_than_days
            )
    
    except KeyboardInterrupt:
        print("\nCleanup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Cleanup failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
