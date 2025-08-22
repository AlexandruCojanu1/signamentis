#!/usr/bin/env python3
"""
SignaMentis Docker Management Scripts
Provides easy commands for managing the Docker infrastructure
"""

import argparse
import subprocess
import sys
import time
import os
from pathlib import Path
from typing import List, Dict, Optional
import yaml
import docker
from docker.errors import DockerException


class DockerManager:
    """Manages Docker containers and services for SignaMentis"""
    
    def __init__(self, compose_file: str = "docker-compose.yml"):
        self.compose_file = compose_file
        self.project_name = "signa_mentis"
        self.client = None
        
        try:
            self.client = docker.from_env()
        except DockerException as e:
            print(f"Warning: Docker client not available: {e}")
    
    def run_command(self, command: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run a shell command"""
        try:
            result = subprocess.run(
                command,
                check=check,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent
            )
            return result
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {' '.join(command)}")
            print(f"Error: {e}")
            if check:
                sys.exit(1)
            return e
    
    def start_services(self, services: Optional[List[str]] = None) -> None:
        """Start all services or specific services"""
        cmd = ["docker-compose", "-f", self.compose_file, "up", "-d"]
        if services:
            cmd.extend(services)
        
        print("Starting services...")
        result = self.run_command(cmd)
        print(result.stdout)
        
        if result.returncode == 0:
            print("✅ Services started successfully!")
            self.wait_for_health_checks()
    
    def stop_services(self, services: Optional[List[str]] = None) -> None:
        """Stop all services or specific services"""
        cmd = ["docker-compose", "-f", self.compose_file, "down"]
        if services:
            cmd = ["docker-compose", "-f", self.compose_file, "stop"] + services
        
        print("Stopping services...")
        result = self.run_command(cmd)
        print(result.stdout)
        
        if result.returncode == 0:
            print("✅ Services stopped successfully!")
    
    def restart_services(self, services: Optional[List[str]] = None) -> None:
        """Restart all services or specific services"""
        self.stop_services(services)
        time.sleep(2)
        self.start_services(services)
    
    def status(self) -> None:
        """Show status of all services"""
        cmd = ["docker-compose", "-f", self.compose_file, "ps"]
        result = self.run_command(cmd)
        print(result.stdout)
    
    def logs(self, service: str, follow: bool = False, tail: int = 100) -> None:
        """Show logs for a specific service"""
        cmd = ["docker-compose", "-f", self.compose_file, "logs"]
        if follow:
            cmd.append("-f")
        if tail:
            cmd.extend(["--tail", str(tail)])
        cmd.append(service)
        
        # For logs, we want to see output in real-time
        try:
            subprocess.run(cmd, cwd=Path(__file__).parent.parent)
        except KeyboardInterrupt:
            print("\nLogs stopped")
    
    def build_services(self, services: Optional[List[str]] = None, no_cache: bool = False) -> None:
        """Build Docker images"""
        cmd = ["docker-compose", "-f", self.compose_file, "build"]
        if no_cache:
            cmd.append("--no-cache")
        if services:
            cmd.extend(services)
        
        print("Building services...")
        result = self.run_command(cmd)
        print(result.stdout)
        
        if result.returncode == 0:
            print("✅ Services built successfully!")
    
    def clean_up(self, volumes: bool = False, images: bool = False) -> None:
        """Clean up Docker resources"""
        cmd = ["docker-compose", "-f", self.compose_file, "down"]
        if volumes:
            cmd.append("--volumes")
        
        print("Cleaning up...")
        result = self.run_command(cmd)
        print(result.stdout)
        
        if images:
            print("Removing unused images...")
            subprocess.run(["docker", "image", "prune", "-f"])
        
        if result.returncode == 0:
            print("✅ Cleanup completed!")
    
    def scale_service(self, service: str, replicas: int) -> None:
        """Scale a service to specified number of replicas"""
        cmd = ["docker-compose", "-f", self.compose_file, "up", "-d", "--scale", f"{service}={replicas}"]
        
        print(f"Scaling {service} to {replicas} replicas...")
        result = self.run_command(cmd)
        print(result.stdout)
        
        if result.returncode == 0:
            print(f"✅ {service} scaled to {replicas} replicas!")
    
    def wait_for_health_checks(self, timeout: int = 300) -> None:
        """Wait for all services to be healthy"""
        print("Waiting for health checks...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            cmd = ["docker-compose", "-f", self.compose_file, "ps"]
            result = self.run_command(cmd, check=False)
            
            if "healthy" in result.stdout and "unhealthy" not in result.stdout:
                print("✅ All services are healthy!")
                return
            
            print("Waiting for services to be healthy...")
            time.sleep(10)
        
        print("⚠️  Timeout waiting for health checks")
    
    def exec_command(self, service: str, command: str) -> None:
        """Execute a command in a running container"""
        cmd = ["docker-compose", "-f", self.compose_file, "exec", service, "bash", "-c", command]
        
        try:
            subprocess.run(cmd, cwd=Path(__file__).parent.parent)
        except KeyboardInterrupt:
            print("\nCommand interrupted")
    
    def backup_data(self, backup_dir: str = "./backups") -> None:
        """Backup data volumes"""
        backup_path = Path(backup_dir)
        backup_path.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Get list of volumes
        cmd = ["docker", "volume", "ls", "-q", "-f", f"name={self.project_name}"]
        result = self.run_command(cmd)
        volumes = result.stdout.strip().split('\n')
        
        for volume in volumes:
            if volume:
                backup_file = backup_path / f"{volume}_{timestamp}.tar"
                print(f"Backing up {volume} to {backup_file}...")
                
                # Create backup
                backup_cmd = [
                    "docker", "run", "--rm", "-v", f"{volume}:/data", "-v", 
                    f"{backup_path}:/backup", "alpine", "tar", "czf", 
                    f"/backup/{backup_file.name}", "-C", "/data", "."
                ]
                
                try:
                    subprocess.run(backup_cmd, check=True)
                    print(f"✅ {volume} backed up successfully!")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Failed to backup {volume}: {e}")
    
    def restore_data(self, backup_file: str) -> None:
        """Restore data from backup"""
        backup_path = Path(backup_file)
        if not backup_path.exists():
            print(f"❌ Backup file not found: {backup_file}")
            return
        
        # Extract volume name from backup filename
        volume_name = backup_path.stem.split('_')[0]
        
        print(f"Restoring {volume_name} from {backup_file}...")
        
        # Stop services first
        self.stop_services()
        
        # Remove existing volume
        subprocess.run(["docker", "volume", "rm", volume_name], check=False)
        
        # Create new volume
        subprocess.run(["docker", "volume", "create", volume_name], check=True)
        
        # Restore data
        restore_cmd = [
            "docker", "run", "--rm", "-v", f"{volume_name}:/data", "-v", 
            f"{backup_path}:/backup", "alpine", "tar", "xzf", 
            f"/backup/{backup_path.name}", "-C", "/data"
        ]
        
        try:
            subprocess.run(restore_cmd, check=True)
            print(f"✅ {volume_name} restored successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to restore {volume_name}: {e}")
    
    def show_metrics(self) -> None:
        """Show system metrics and resource usage"""
        if not self.client:
            print("❌ Docker client not available")
            return
        
        print("📊 System Metrics:")
        print("=" * 50)
        
        # Container stats
        containers = self.client.containers.list()
        total_cpu = 0
        total_memory = 0
        
        for container in containers:
            if container.name.startswith(self.project_name):
                stats = container.stats(stream=False)
                
                # Calculate CPU usage
                cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                           stats['precpu_stats']['cpu_usage']['total_usage']
                system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                             stats['precpu_stats']['system_cpu_usage']
                
                if system_delta > 0:
                    cpu_percent = (cpu_delta / system_delta) * 100
                    total_cpu += cpu_percent
                
                # Memory usage
                memory_usage = stats['memory_stats']['usage'] / (1024 * 1024)  # MB
                total_memory += memory_usage
                
                print(f"{container.name}:")
                print(f"  CPU: {cpu_percent:.1f}%")
                print(f"  Memory: {memory_usage:.1f} MB")
                print(f"  Status: {container.status}")
                print()
        
        print(f"Total CPU: {total_cpu:.1f}%")
        print(f"Total Memory: {total_memory:.1f} MB")


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="SignaMentis Docker Management")
    parser.add_argument("command", choices=[
        "start", "stop", "restart", "status", "logs", "build", 
        "clean", "scale", "exec", "backup", "restore", "metrics"
    ])
    parser.add_argument("--services", nargs="+", help="Specific services to operate on")
    parser.add_argument("--service", help="Service name for logs/exec commands")
    parser.add_argument("--follow", "-f", action="store_true", help="Follow logs")
    parser.add_argument("--tail", "-t", type=int, default=100, help="Number of log lines to show")
    parser.add_argument("--no-cache", action="store_true", help="Build without cache")
    parser.add_argument("--volumes", action="store_true", help="Remove volumes on cleanup")
    parser.add_argument("--images", action="store_true", help="Remove images on cleanup")
    parser.add_argument("--replicas", type=int, help="Number of replicas for scaling")
    parser.add_argument("--command", help="Command to execute in container")
    parser.add_argument("--backup-file", help="Backup file for restore")
    parser.add_argument("--backup-dir", default="./backups", help="Backup directory")
    
    args = parser.parse_args()
    
    manager = DockerManager()
    
    try:
        if args.command == "start":
            manager.start_services(args.services)
        elif args.command == "stop":
            manager.stop_services(args.services)
        elif args.command == "restart":
            manager.restart_services(args.services)
        elif args.command == "status":
            manager.status()
        elif args.command == "logs":
            if not args.service:
                print("❌ Service name required for logs command")
                sys.exit(1)
            manager.logs(args.service, args.follow, args.tail)
        elif args.command == "build":
            manager.build_services(args.services, args.no_cache)
        elif args.command == "clean":
            manager.clean_up(args.volumes, args.images)
        elif args.command == "scale":
            if not args.service or not args.replicas:
                print("❌ Service name and replicas required for scale command")
                sys.exit(1)
            manager.scale_service(args.service, args.replicas)
        elif args.command == "exec":
            if not args.service or not args.command:
                print("❌ Service name and command required for exec command")
                sys.exit(1)
            manager.exec_command(args.service, args.command)
        elif args.command == "backup":
            manager.backup_data(args.backup_dir)
        elif args.command == "restore":
            if not args.backup_file:
                print("❌ Backup file required for restore command")
                sys.exit(1)
            manager.restore_data(args.backup_file)
        elif args.command == "metrics":
            manager.show_metrics()
    
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
