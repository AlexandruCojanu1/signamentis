#!/usr/bin/env python3
"""
SignaMentis Deployment Test Script
Comprehensive testing of all deployed services and components
"""

import argparse
import subprocess
import sys
import time
import requests
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import docker
from docker.errors import DockerException


class DeploymentTester:
    """Comprehensive testing of SignaMentis deployment"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_results = {}
        self.client = None
        
        try:
            self.client = docker.from_env()
        except DockerException as e:
            print(f"Warning: Docker client not available: {e}")
    
    def log(self, message: str, level: str = "INFO") -> None:
        """Log test messages"""
        timestamp = datetime.now().strftime("%H:%M:%S")
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
    
    def test_docker_services(self) -> Dict[str, bool]:
        """Test Docker services status"""
        self.log("Testing Docker services...")
        
        results = {}
        
        try:
            # Check if docker-compose is running
            compose_result = self.run_command(["docker", "compose", "ps"], check=False)
            
            if compose_result.returncode != 0:
                self.log("❌ Docker Compose not running", "ERROR")
                return {"docker_compose": False}
            
            # Parse service status
            lines = compose_result.stdout.strip().split('\n')
            for line in lines[2:]:  # Skip header lines
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 3:
                        service_name = parts[0]
                        status = parts[2]
                        is_healthy = "healthy" in status
                        results[service_name] = is_healthy
                        
                        if is_healthy:
                            self.log(f"✅ {service_name}: {status}")
                        else:
                            self.log(f"❌ {service_name}: {status}", "ERROR")
            
            return results
            
        except Exception as e:
            self.log(f"❌ Docker services test failed: {e}", "ERROR")
            return {"docker_services": False}
    
    def test_health_endpoints(self) -> Dict[str, bool]:
        """Test health check endpoints"""
        self.log("Testing health endpoints...")
        
        endpoints = {
            "trading_system": "http://localhost:8000/health",
            "news_nlp": "http://localhost:8001/health",
            "mlflow": "http://localhost:5000/health",
            "grafana": "http://localhost:3000/api/health",
            "prometheus": "http://localhost:9090/-/healthy"
        }
        
        results = {}
        
        for service, url in endpoints.items():
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    self.log(f"✅ {service}: Healthy")
                    results[service] = True
                else:
                    self.log(f"❌ {service}: Status {response.status_code}", "ERROR")
                    results[service] = False
            except Exception as e:
                self.log(f"❌ {service}: Connection failed - {e}", "ERROR")
                results[service] = False
        
        return results
    
    def test_api_endpoints(self) -> Dict[str, bool]:
        """Test API functionality"""
        self.log("Testing API endpoints...")
        
        # Test trading system API
        trading_endpoints = [
            ("/api/v1/status", "GET"),
            ("/api/v1/portfolio", "GET"),
            ("/api/v1/signals", "GET")
        ]
        
        results = {}
        
        for endpoint, method in trading_endpoints:
            try:
                url = f"http://localhost:8000{endpoint}"
                if method == "GET":
                    response = requests.get(url, timeout=10)
                else:
                    response = requests.post(url, timeout=10)
                
                if response.status_code in [200, 401, 403]:  # 401/403 means endpoint exists but needs auth
                    self.log(f"✅ {endpoint}: Accessible")
                    results[f"trading_{endpoint}"] = True
                else:
                    self.log(f"❌ {endpoint}: Status {response.status_code}", "ERROR")
                    results[f"trading_{endpoint}"] = False
                    
            except Exception as e:
                self.log(f"❌ {endpoint}: Failed - {e}", "ERROR")
                results[f"trading_{endpoint}"] = False
        
        # Test news NLP API
        news_endpoints = [
            ("/api/v1/news", "GET"),
            ("/api/v1/sentiment", "GET")
        ]
        
        for endpoint, method in news_endpoints:
            try:
                url = f"http://localhost:8001{endpoint}"
                if method == "GET":
                    response = requests.get(url, timeout=10)
                else:
                    response = requests.post(url, timeout=10)
                
                if response.status_code in [200, 401, 403]:
                    self.log(f"✅ {endpoint}: Accessible")
                    results[f"news_{endpoint}"] = True
                else:
                    self.log(f"❌ {endpoint}: Status {response.status_code}", "ERROR")
                    results[f"news_{endpoint}"] = False
                    
            except Exception as e:
                self.log(f"❌ {endpoint}: Failed - {e}", "ERROR")
                results[f"news_{endpoint}"] = False
        
        return results
    
    def test_database_connections(self) -> Dict[str, bool]:
        """Test database connectivity"""
        self.log("Testing database connections...")
        
        results = {}
        
        # Test MongoDB
        try:
            mongo_result = self.run_command([
                "docker", "compose", "exec", "-T", "mongodb", 
                "mongosh", "--eval", "db.adminCommand('ping')"
            ], check=False)
            
            if mongo_result.returncode == 0:
                self.log("✅ MongoDB: Connected")
                results["mongodb"] = True
            else:
                self.log("❌ MongoDB: Connection failed", "ERROR")
                results["mongodb"] = False
        except Exception as e:
            self.log(f"❌ MongoDB test failed: {e}", "ERROR")
            results["mongodb"] = False
        
        # Test Redis
        try:
            redis_result = self.run_command([
                "docker", "compose", "exec", "-T", "redis", 
                "redis-cli", "ping"
            ], check=False)
            
            if redis_result.returncode == 0 and "PONG" in redis_result.stdout:
                self.log("✅ Redis: Connected")
                results["redis"] = True
            else:
                self.log("❌ Redis: Connection failed", "ERROR")
                results["redis"] = False
        except Exception as e:
            self.log(f"❌ Redis test failed: {e}", "ERROR")
            results["redis"] = False
        
        # Test RabbitMQ
        try:
            rabbit_result = self.run_command([
                "docker", "compose", "exec", "-T", "rabbitmq", 
                "rabbitmq-diagnostics", "ping"
            ], check=False)
            
            if rabbit_result.returncode == 0:
                self.log("✅ RabbitMQ: Connected")
                results["rabbitmq"] = True
            else:
                self.log("❌ RabbitMQ: Connection failed", "ERROR")
                results["rabbitmq"] = False
        except Exception as e:
            self.log(f"❌ RabbitMQ test failed: {e}", "ERROR")
            results["rabbitmq"] = False
        
        return results
    
    def test_mlflow_functionality(self) -> Dict[str, bool]:
        """Test MLflow functionality"""
        self.log("Testing MLflow functionality...")
        
        results = {}
        
        try:
            # Test MLflow API
            response = requests.get("http://localhost:5000/api/2.0/mlflow/experiments/list", timeout=10)
            
            if response.status_code == 200:
                self.log("✅ MLflow API: Accessible")
                results["mlflow_api"] = True
            else:
                self.log(f"❌ MLflow API: Status {response.status_code}", "ERROR")
                results["mlflow_api"] = False
                
        except Exception as e:
            self.log(f"❌ MLflow API test failed: {e}", "ERROR")
            results["mlflow_api"] = False
        
        # Test MinIO connection (MLflow artifact store)
        try:
            minio_result = self.run_command([
                "docker", "compose", "exec", "-T", "minio", 
                "mc", "config", "host", "add", "local", "http://localhost:9000", "minioadmin", "minioadmin"
            ], check=False)
            
            if minio_result.returncode == 0:
                self.log("✅ MinIO: Accessible")
                results["minio"] = True
            else:
                self.log("❌ MinIO: Connection failed", "ERROR")
                results["minio"] = False
                
        except Exception as e:
            self.log(f"❌ MinIO test failed: {e}", "ERROR")
            results["minio"] = False
        
        return results
    
    def test_monitoring_stack(self) -> Dict[str, bool]:
        """Test monitoring and observability"""
        self.log("Testing monitoring stack...")
        
        results = {}
        
        # Test Prometheus
        try:
            response = requests.get("http://localhost:9090/api/v1/query?query=up", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    self.log("✅ Prometheus: API responding")
                    results["prometheus"] = True
                else:
                    self.log("❌ Prometheus: API error", "ERROR")
                    results["prometheus"] = False
            else:
                self.log(f"❌ Prometheus: Status {response.status_code}", "ERROR")
                results["prometheus"] = False
                
        except Exception as e:
            self.log(f"❌ Prometheus test failed: {e}", "ERROR")
            results["prometheus"] = False
        
        # Test Grafana
        try:
            response = requests.get("http://localhost:3000/api/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("database") == "ok":
                    self.log("✅ Grafana: Healthy")
                    results["grafana"] = True
                else:
                    self.log("❌ Grafana: Database issues", "ERROR")
                    results["grafana"] = False
            else:
                self.log(f"❌ Grafana: Status {response.status_code}", "ERROR")
                results["grafana"] = False
                
        except Exception as e:
            self.log(f"❌ Grafana test failed: {e}", "ERROR")
            results["grafana"] = False
        
        return results
    
    def test_celery_workers(self) -> Dict[str, bool]:
        """Test Celery background workers"""
        self.log("Testing Celery workers...")
        
        results = {}
        
        try:
            # Check if workers are running
            worker_result = self.run_command([
                "docker", "compose", "exec", "-T", "celery_worker", 
                "celery", "-A", "scripts.celery_app", "inspect", "active"
            ], check=False)
            
            if worker_result.returncode == 0:
                self.log("✅ Celery Worker: Active")
                results["celery_worker"] = True
            else:
                self.log("❌ Celery Worker: Not responding", "ERROR")
                results["celery_worker"] = False
                
        except Exception as e:
            self.log(f"❌ Celery Worker test failed: {e}", "ERROR")
            results["celery_worker"] = False
        
        # Test Flower monitoring
        try:
            response = requests.get("http://localhost:5555/", timeout=10)
            
            if response.status_code == 200:
                self.log("✅ Flower: Accessible")
                results["flower"] = True
            else:
                self.log(f"❌ Flower: Status {response.status_code}", "ERROR")
                results["flower"] = False
                
        except Exception as e:
            self.log(f"❌ Flower test failed: {e}", "ERROR")
            results["flower"] = False
        
        return results
    
    def test_network_connectivity(self) -> Dict[str, bool]:
        """Test network connectivity between services"""
        self.log("Testing network connectivity...")
        
        results = {}
        
        # Test internal network
        try:
            network_result = self.run_command([
                "docker", "network", "inspect", "signa_mentis_network"
            ], check=False)
            
            if network_result.returncode == 0:
                self.log("✅ Internal Network: Configured")
                results["internal_network"] = True
            else:
                self.log("❌ Internal Network: Not found", "ERROR")
                results["internal_network"] = False
                
        except Exception as e:
            self.log(f"❌ Network test failed: {e}", "ERROR")
            results["internal_network"] = False
        
        # Test service communication
        try:
            # Test if trading system can reach MongoDB
            ping_result = self.run_command([
                "docker", "compose", "exec", "-T", "trading_system", 
                "ping", "-c", "1", "mongodb"
            ], check=False)
            
            if ping_result.returncode == 0:
                self.log("✅ Service Communication: Working")
                results["service_communication"] = True
            else:
                self.log("❌ Service Communication: Failed", "ERROR")
                results["service_communication"] = False
                
        except Exception as e:
            self.log(f"❌ Service communication test failed: {e}", "ERROR")
            results["service_communication"] = False
        
        return results
    
    def test_data_quality(self) -> Dict[str, bool]:
        """Test data quality checks"""
        self.log("Testing data quality...")
        
        results = {}
        
        try:
            # Run Great Expectations
            gx_result = self.run_command([
                "python", "qa/gx/expectations.py"
            ], check=False)
            
            if gx_result.returncode == 0:
                self.log("✅ Data Quality: Passed")
                results["data_quality"] = True
            else:
                self.log("❌ Data Quality: Failed", "ERROR")
                results["data_quality"] = False
                
        except Exception as e:
            self.log(f"❌ Data quality test failed: {e}", "ERROR")
            results["data_quality"] = False
        
        return results
    
    def run_all_tests(self) -> Dict[str, Dict[str, bool]]:
        """Run all deployment tests"""
        self.log("🚀 Starting comprehensive deployment testing...")
        self.log("=" * 60)
        
        all_results = {}
        
        # Run all test suites
        test_suites = [
            ("Docker Services", self.test_docker_services),
            ("Health Endpoints", self.test_health_endpoints),
            ("API Endpoints", self.test_api_endpoints),
            ("Database Connections", self.test_database_connections),
            ("MLflow Functionality", self.test_mlflow_functionality),
            ("Monitoring Stack", self.test_monitoring_stack),
            ("Celery Workers", self.test_celery_workers),
            ("Network Connectivity", self.test_network_connectivity),
            ("Data Quality", self.test_data_quality)
        ]
        
        for suite_name, test_function in test_suites:
            self.log(f"\n🔍 Testing: {suite_name}")
            self.log("-" * 40)
            
            try:
                results = test_function()
                all_results[suite_name] = results
                
                # Calculate success rate
                total_tests = len(results)
                passed_tests = sum(1 for result in results.values() if result)
                success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
                
                self.log(f"📊 {suite_name}: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
                
            except Exception as e:
                self.log(f"❌ {suite_name} test suite failed: {e}", "ERROR")
                all_results[suite_name] = {"test_suite": False}
        
        return all_results
    
    def generate_report(self, results: Dict[str, Dict[str, bool]]) -> None:
        """Generate comprehensive test report"""
        self.log("\n📋 Generating Test Report...")
        self.log("=" * 60)
        
        # Calculate overall statistics
        total_tests = 0
        total_passed = 0
        
        for suite_name, suite_results in results.items():
            suite_total = len(suite_results)
            suite_passed = sum(1 for result in suite_results.values() if result)
            
            total_tests += suite_total
            total_passed += suite_passed
            
            # Show suite results
            status = "✅ PASS" if suite_passed == suite_total else "❌ FAIL"
            self.log(f"{status} {suite_name}: {suite_passed}/{suite_total}")
        
        # Overall results
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        self.log(f"\n🎯 Overall Results: {total_passed}/{total_tests} tests passed ({overall_success_rate:.1f}%)")
        
        if overall_success_rate >= 90:
            self.log("🎉 Excellent! Deployment is working perfectly!", "SUCCESS")
        elif overall_success_rate >= 75:
            self.log("👍 Good! Most components are working correctly.", "SUCCESS")
        elif overall_success_rate >= 50:
            self.log("⚠️  Fair! Some components have issues that need attention.", "WARNING")
        else:
            self.log("❌ Poor! Many components are not working correctly.", "ERROR")
        
        # Detailed results
        self.log("\n📊 Detailed Results:")
        self.log("-" * 40)
        
        for suite_name, suite_results in results.items():
            self.log(f"\n{suite_name}:")
            for test_name, test_result in suite_results.items():
                status = "✅" if test_result else "❌"
                self.log(f"  {status} {test_name}")
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.project_root / "logs" / f"deployment_test_report_{timestamp}.json"
        report_file.parent.mkdir(exist_ok=True)
        
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "overall_results": {
                "total_tests": total_tests,
                "total_passed": total_passed,
                "success_rate": overall_success_rate
            },
            "detailed_results": results
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        self.log(f"\n📄 Detailed report saved to: {report_file}")
        
        return overall_success_rate >= 75  # Return True if deployment is considered successful


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="SignaMentis Deployment Tester")
    parser.add_argument("--suite", choices=[
        "all", "docker", "health", "api", "database", "mlflow", 
        "monitoring", "celery", "network", "data-quality"
    ], default="all", help="Test suite to run")
    parser.add_argument("--save-report", action="store_true", help="Save detailed report to file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    tester = DeploymentTester()
    
    try:
        if args.suite == "all":
            results = tester.run_all_tests()
            success = tester.generate_report(results)
            
            if args.save_report:
                # Report is automatically saved
                pass
            
            sys.exit(0 if success else 1)
        
        else:
            # Run specific test suite
            test_functions = {
                "docker": tester.test_docker_services,
                "health": tester.test_health_endpoints,
                "api": tester.test_api_endpoints,
                "database": tester.test_database_connections,
                "mlflow": tester.test_mlflow_functionality,
                "monitoring": tester.test_monitoring_stack,
                "celery": tester.test_celery_workers,
                "network": tester.test_network_connectivity,
                "data-quality": tester.test_data_quality
            }
            
            if args.suite in test_functions:
                tester.log(f"Running {args.suite} tests...")
                results = test_functions[args.suite]()
                
                # Show results
                total = len(results)
                passed = sum(1 for result in results.values() if result)
                success_rate = (passed / total * 100) if total > 0 else 0
                
                tester.log(f"Results: {passed}/{total} tests passed ({success_rate:.1f}%)")
                
                for test_name, test_result in results.items():
                    status = "✅" if test_result else "❌"
                    tester.log(f"{status} {test_name}")
                
                sys.exit(0 if passed == total else 1)
            else:
                tester.log(f"Unknown test suite: {args.suite}", "ERROR")
                sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Testing failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
