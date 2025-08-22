#!/usr/bin/env python3
"""
SignaMentis Automated Deployment Script
Handles complete deployment pipeline including testing, building, and deployment
"""

import argparse
import subprocess
import sys
import time
import os
from pathlib import Path
from typing import List, Dict, Optional
import yaml
import json
from datetime import datetime


class DeployManager:
    """Manages the complete deployment pipeline for SignaMentis"""
    
    def __init__(self, config_file: str = "deployment_config.yml"):
        self.config_file = config_file
        self.config = self.load_config()
        self.project_root = Path(__file__).parent.parent
        self.deployment_log = []
        
    def load_config(self) -> Dict:
        """Load deployment configuration"""
        config_path = Path(__file__).parent / self.config_file
        
        if not config_path.exists():
            # Create default config
            default_config = {
                "environment": "development",
                "services": {
                    "trading_system": {"port": 8000, "health_check": "/health"},
                    "news_nlp": {"port": 8001, "health_check": "/health"},
                    "mlflow": {"port": 5000, "health_check": "/health"},
                    "grafana": {"port": 3000, "health_check": "/api/health"}
                },
                "deployment": {
                    "backup_before_deploy": True,
                    "run_tests": True,
                    "health_check_timeout": 300,
                    "rollback_on_failure": True
                },
                "monitoring": {
                    "prometheus": {"port": 9090},
                    "grafana": {"port": 3000}
                }
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            
            return default_config
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def log(self, message: str, level: str = "INFO") -> None:
        """Log deployment messages"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        self.deployment_log.append(log_entry)
        print(log_entry)
    
    def run_command(self, command: List[str], check: bool = True, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
        """Run a shell command"""
        try:
            result = subprocess.run(
                command,
                check=check,
                capture_output=True,
                text=True,
                cwd=cwd or self.project_root
            )
            return result
        except subprocess.CalledProcessError as e:
            self.log(f"Command failed: {' '.join(command)}", "ERROR")
            self.log(f"Error: {e}", "ERROR")
            if check:
                raise
            return e
    
    def run_tests(self) -> bool:
        """Run all tests before deployment"""
        if not self.config["deployment"]["run_tests"]:
            self.log("Skipping tests as configured", "WARNING")
            return True
        
        self.log("Running tests...")
        
        # Run unit tests
        test_result = self.run_command(["python", "-m", "pytest", "tests/", "-v"], check=False)
        if test_result.returncode != 0:
            self.log("❌ Tests failed!", "ERROR")
            return False
        
        # Run property tests
        property_test_result = self.run_command(["python", "tests/property_tests.py"], check=False)
        if property_test_result.returncode != 0:
            self.log("❌ Property tests failed!", "ERROR")
            return False
        
        # Run data quality checks
        gx_result = self.run_command(["python", "qa/gx/expectations.py"], check=False)
        if gx_result.returncode != 0:
            self.log("❌ Data quality checks failed!", "ERROR")
            return False
        
        self.log("✅ All tests passed!")
        return True
    
    def backup_data(self) -> bool:
        """Backup data before deployment"""
        if not self.config["deployment"]["backup_before_deploy"]:
            self.log("Skipping backup as configured", "WARNING")
            return True
        
        self.log("Creating backup...")
        
        backup_dir = self.project_root / "backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup Docker volumes
        backup_result = self.run_command([
            "python", "scripts/docker_management.py", "backup", 
            "--backup-dir", str(backup_dir)
        ], check=False)
        
        if backup_result.returncode != 0:
            self.log("❌ Backup failed!", "ERROR")
            return False
        
        self.log(f"✅ Backup created in {backup_dir}")
        return True
    
    def build_images(self, no_cache: bool = False) -> bool:
        """Build Docker images"""
        self.log("Building Docker images...")
        
        build_result = self.run_command([
            "python", "scripts/docker_management.py", "build"
        ] + (["--no-cache"] if no_cache else []), check=False)
        
        if build_result.returncode != 0:
            self.log("❌ Build failed!", "ERROR")
            return False
        
        self.log("✅ Images built successfully!")
        return True
    
    def deploy_services(self) -> bool:
        """Deploy all services"""
        self.log("Deploying services...")
        
        # Stop existing services
        self.log("Stopping existing services...")
        stop_result = self.run_command([
            "python", "scripts/docker_management.py", "stop"
        ], check=False)
        
        # Start services
        self.log("Starting services...")
        start_result = self.run_command([
            "python", "scripts/docker_management.py", "start"
        ], check=False)
        
        if start_result.returncode != 0:
            self.log("❌ Service deployment failed!", "ERROR")
            return False
        
        self.log("✅ Services deployed successfully!")
        return True
    
    def wait_for_health_checks(self) -> bool:
        """Wait for all services to be healthy"""
        timeout = self.config["deployment"]["health_check_timeout"]
        self.log(f"Waiting for health checks (timeout: {timeout}s)...")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check service status
            status_result = self.run_command([
                "python", "scripts/docker_management.py", "status"
            ], check=False)
            
            if "healthy" in status_result.stdout and "unhealthy" not in status_result.stdout:
                self.log("✅ All services are healthy!")
                return True
            
            self.log("Waiting for services to be healthy...")
            time.sleep(10)
        
        self.log("⚠️  Health check timeout", "WARNING")
        return False
    
    def run_smoke_tests(self) -> bool:
        """Run smoke tests after deployment"""
        self.log("Running smoke tests...")
        
        # Test main API endpoints
        endpoints = [
            ("http://localhost:8000/health", "Trading System"),
            ("http://localhost:8001/health", "News NLP"),
            ("http://localhost:5000/health", "MLflow"),
            ("http://localhost:3000/api/health", "Grafana")
        ]
        
        for endpoint, service_name in endpoints:
            try:
                import requests
                response = requests.get(endpoint, timeout=10)
                if response.status_code == 200:
                    self.log(f"✅ {service_name} is responding")
                else:
                    self.log(f"❌ {service_name} returned status {response.status_code}", "ERROR")
                    return False
            except Exception as e:
                self.log(f"❌ {service_name} test failed: {e}", "ERROR")
                return False
        
        self.log("✅ All smoke tests passed!")
        return True
    
    def rollback(self) -> bool:
        """Rollback to previous version on failure"""
        if not self.config["deployment"]["rollback_on_failure"]:
            self.log("Rollback disabled by configuration", "WARNING")
            return False
        
        self.log("Rolling back deployment...")
        
        # Stop current services
        self.run_command(["python", "scripts/docker_management.py", "stop"], check=False)
        
        # Restore from backup if available
        backup_dir = self.project_root / "backups"
        if backup_dir.exists():
            backups = sorted(backup_dir.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)
            if len(backups) > 1:  # Skip current backup
                latest_backup = backups[1]  # Get previous backup
                self.log(f"Restoring from backup: {latest_backup.name}")
                
                restore_result = self.run_command([
                    "python", "scripts/docker_management.py", "restore",
                    "--backup-file", str(latest_backup)
                ], check=False)
                
                if restore_result.returncode == 0:
                    self.log("✅ Rollback completed successfully!")
                    return True
        
        self.log("❌ Rollback failed!", "ERROR")
        return False
    
    def deploy(self, environment: str = None, no_cache: bool = False) -> bool:
        """Complete deployment pipeline"""
        if environment:
            self.config["environment"] = environment
        
        self.log(f"Starting deployment to {self.config['environment']} environment")
        self.log("=" * 60)
        
        try:
            # Step 1: Run tests
            if not self.run_tests():
                return False
            
            # Step 2: Create backup
            if not self.backup_data():
                return False
            
            # Step 3: Build images
            if not self.build_images(no_cache):
                return False
            
            # Step 4: Deploy services
            if not self.deploy_services():
                return False
            
            # Step 5: Wait for health checks
            if not self.wait_for_health_checks():
                self.log("⚠️  Health checks failed, attempting rollback...", "WARNING")
                if self.rollback():
                    return False
                else:
                    self.log("❌ Rollback failed!", "ERROR")
                    return False
            
            # Step 6: Run smoke tests
            if not self.run_smoke_tests():
                self.log("⚠️  Smoke tests failed, attempting rollback...", "WARNING")
                if self.rollback():
                    return False
                else:
                    self.log("❌ Rollback failed!", "ERROR")
                    return False
            
            self.log("=" * 60)
            self.log("🎉 Deployment completed successfully!")
            return True
            
        except Exception as e:
            self.log(f"❌ Deployment failed with error: {e}", "ERROR")
            self.log("Attempting rollback...", "WARNING")
            self.rollback()
            return False
    
    def show_deployment_log(self) -> None:
        """Display deployment log"""
        print("\n📋 Deployment Log:")
        print("=" * 60)
        for entry in self.deployment_log:
            print(entry)
    
    def save_deployment_log(self, filename: str = None) -> None:
        """Save deployment log to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"deployment_log_{timestamp}.txt"
        
        log_path = self.project_root / "logs" / filename
        log_path.parent.mkdir(exist_ok=True)
        
        with open(log_path, 'w') as f:
            for entry in self.deployment_log:
                f.write(entry + '\n')
        
        self.log(f"Deployment log saved to {log_path}")


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="SignaMentis Deployment Manager")
    parser.add_argument("command", choices=["deploy", "test", "build", "status", "rollback", "logs"])
    parser.add_argument("--environment", "-e", choices=["development", "staging", "production"], 
                       help="Deployment environment")
    parser.add_argument("--no-cache", action="store_true", help="Build without cache")
    parser.add_argument("--config", default="deployment_config.yml", help="Configuration file")
    parser.add_argument("--save-log", action="store_true", help="Save deployment log to file")
    
    args = parser.parse_args()
    
    deploy_manager = DeployManager(args.config)
    
    try:
        if args.command == "deploy":
            success = deploy_manager.deploy(args.environment, args.no_cache)
            if args.save_log:
                deploy_manager.save_deployment_log()
            sys.exit(0 if success else 1)
        
        elif args.command == "test":
            success = deploy_manager.run_tests()
            sys.exit(0 if success else 1)
        
        elif args.command == "build":
            success = deploy_manager.build_images(args.no_cache)
            sys.exit(0 if success else 1)
        
        elif args.command == "status":
            deploy_manager.run_command(["python", "scripts/docker_management.py", "status"])
        
        elif args.command == "rollback":
            success = deploy_manager.rollback()
            sys.exit(0 if success else 1)
        
        elif args.command == "logs":
            deploy_manager.show_deployment_log()
    
    except KeyboardInterrupt:
        print("\nDeployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
