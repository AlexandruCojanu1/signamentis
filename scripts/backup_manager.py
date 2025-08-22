#!/usr/bin/env python3
"""
SignaMentis Backup Manager
Automated backup and restore of data, configurations, and system state
"""

import argparse
import subprocess
import sys
import time
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import tarfile
import gzip
import docker
from docker.errors import DockerException


class BackupManager:
    """Manages automated backup and restore operations for SignaMentis"""
    
    def __init__(self, backup_dir: str = "./backups"):
        self.project_root = Path(__file__).parent.parent
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.client = None
        
        try:
            self.client = docker.from_env()
        except DockerException as e:
            print(f"Warning: Docker client not available: {e}")
    
    def log(self, message: str, level: str = "INFO") -> None:
        """Log backup messages"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
    
    def create_backup(self, backup_name: str = None, include_volumes: bool = True, 
                      include_configs: bool = True, include_data: bool = True,
                      compression: str = "gzip") -> str:
        """Create a comprehensive backup"""
        if not backup_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"signa_mentis_backup_{timestamp}"
        
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(exist_ok=True)
        
        self.log(f"Creating backup: {backup_name}")
        self.log(f"Backup location: {backup_path}")
        
        try:
            # Create backup metadata
            metadata = {
                "backup_name": backup_name,
                "created_at": datetime.now().isoformat(),
                "signa_mentis_version": "2.0.0",
                "components": []
            }
            
            # Backup Docker volumes
            if include_volumes:
                self.log("Backing up Docker volumes...")
                volumes_backup = self._backup_volumes(backup_path)
                metadata["components"].append({
                    "type": "docker_volumes",
                    "status": "completed" if volumes_backup else "failed",
                    "files": volumes_backup
                })
            
            # Backup configurations
            if include_configs:
                self.log("Backing up configurations...")
                configs_backup = self._backup_configs(backup_path)
                metadata["components"].append({
                    "type": "configurations",
                    "status": "completed" if configs_backup else "failed",
                    "files": configs_backup
                })
            
            # Backup data directories
            if include_data:
                self.log("Backing up data directories...")
                data_backup = self._backup_data_dirs(backup_path)
                metadata["components"].append({
                    "type": "data_directories",
                    "status": "completed" if data_backup else "failed",
                    "files": data_backup
                })
            
            # Save metadata
            metadata_file = backup_path / "backup_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Create compressed archive
            archive_name = f"{backup_name}.tar.gz"
            archive_path = self.backup_dir / archive_name
            
            self.log(f"Creating compressed archive: {archive_name}")
            
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(backup_path, arcname=backup_name)
            
            # Remove temporary backup directory
            shutil.rmtree(backup_path)
            
            # Verify archive
            if archive_path.exists():
                archive_size = archive_path.stat().st_size / (1024 * 1024)  # MB
                self.log(f"✅ Backup completed successfully!")
                self.log(f"📦 Archive: {archive_name}")
                self.log(f"💾 Size: {archive_size:.2f} MB")
                
                # Update metadata with final info
                metadata["archive_file"] = archive_name
                metadata["archive_size_mb"] = archive_size
                metadata["final_status"] = "completed"
                
                # Save final metadata
                final_metadata_file = self.backup_dir / f"{backup_name}_metadata.json"
                with open(final_metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                return str(archive_path)
            else:
                self.log("❌ Backup archive creation failed!", "ERROR")
                return None
                
        except Exception as e:
            self.log(f"❌ Backup creation failed: {e}", "ERROR")
            # Cleanup on failure
            if backup_path.exists():
                shutil.rmtree(backup_path)
            return None
    
    def _backup_volumes(self, backup_path: Path) -> List[str]:
        """Backup Docker volumes"""
        volumes_dir = backup_path / "volumes"
        volumes_dir.mkdir(exist_ok=True)
        
        backed_up_files = []
        
        try:
            # Get list of volumes
            volumes_result = self.run_command(["docker", "volume", "ls", "-q", "-f", "name=signa_mentis"], check=False)
            
            if volumes_result.returncode == 0:
                volumes = volumes_result.stdout.strip().split('\n')
                
                for volume in volumes:
                    if volume:
                        volume_backup_file = volumes_dir / f"{volume}.tar"
                        
                        # Create volume backup
                        backup_cmd = [
                            "docker", "run", "--rm", "-v", f"{volume}:/data", 
                            "-v", f"{volumes_dir}:/backup", "alpine", 
                            "tar", "cf", f"/backup/{volume}.tar", "-C", "/data", "."
                        ]
                        
                        result = self.run_command(backup_cmd, check=False)
                        if result.returncode == 0:
                            self.log(f"✅ Volume {volume} backed up")
                            backed_up_files.append(str(volume_backup_file))
                        else:
                            self.log(f"❌ Volume {volume} backup failed", "ERROR")
            
            return backed_up_files
            
        except Exception as e:
            self.log(f"❌ Volume backup failed: {e}", "ERROR")
            return []
    
    def _backup_configs(self, backup_path: Path) -> List[str]:
        """Backup configuration files"""
        configs_dir = backup_path / "configs"
        configs_dir.mkdir(exist_ok=True)
        
        backed_up_files = []
        
        try:
            # Configuration files to backup
            config_files = [
                "docker-compose.yml",
                "Dockerfile",
                ".env",
                "requirements.txt"
            ]
            
            # Configuration directories to backup
            config_dirs = [
                "docker/",
                "config/",
                "scripts/"
            ]
            
            # Backup individual config files
            for config_file in config_files:
                file_path = self.project_root / config_file
                if file_path.exists():
                    shutil.copy2(file_path, configs_dir)
                    backed_up_files.append(str(configs_dir / config_file))
                    self.log(f"✅ Config file {config_file} backed up")
            
            # Backup config directories
            for config_dir in config_dirs:
                dir_path = self.project_root / config_dir
                if dir_path.exists():
                    shutil.copytree(dir_path, configs_dir / config_dir, dirs_exist_ok=True)
                    backed_up_files.append(str(configs_dir / config_dir))
                    self.log(f"✅ Config directory {config_dir} backed up")
            
            return backed_up_files
            
        except Exception as e:
            self.log(f"❌ Config backup failed: {e}", "ERROR")
            return []
    
    def _backup_data_dirs(self, backup_path: Path) -> List[str]:
        """Backup data directories"""
        data_dir = backup_path / "data"
        data_dir.mkdir(exist_ok=True)
        
        backed_up_files = []
        
        try:
            # Data directories to backup
            data_dirs = ["data", "models", "logs", "cache"]
            
            for dir_name in data_dirs:
                dir_path = self.project_root / dir_name
                if dir_path.exists():
                    # Create tar archive of data directory
                    tar_file = data_dir / f"{dir_name}.tar"
                    
                    with tarfile.open(tar_file, "w") as tar:
                        tar.add(dir_path, arcname=dir_name)
                    
                    backed_up_files.append(str(tar_file))
                    self.log(f"✅ Data directory {dir_name} backed up")
            
            return backed_up_files
            
        except Exception as e:
            self.log(f"❌ Data backup failed: {e}", "ERROR")
            return []
    
    def restore_backup(self, backup_file: str, restore_volumes: bool = True,
                      restore_configs: bool = True, restore_data: bool = True,
                      force: bool = False) -> bool:
        """Restore from backup"""
        backup_path = Path(backup_file)
        
        if not backup_path.exists():
            self.log(f"❌ Backup file not found: {backup_file}", "ERROR")
            return False
        
        if not force:
            response = input(f"⚠️  This will restore from backup: {backup_file}. Continue? (yes/no): ")
            if response.lower() != "yes":
                self.log("Restore cancelled by user", "WARNING")
                return False
        
        self.log(f"Restoring from backup: {backup_file}")
        
        try:
            # Create temporary restore directory
            restore_dir = self.backup_dir / f"restore_temp_{int(time.time())}"
            restore_dir.mkdir(exist_ok=True)
            
            # Extract backup
            self.log("Extracting backup...")
            with tarfile.open(backup_path, "r:gz") as tar:
                tar.extractall(restore_dir)
            
            # Find the backup directory
            backup_dirs = [d for d in restore_dir.iterdir() if d.is_dir()]
            if not backup_dirs:
                self.log("❌ Invalid backup format", "ERROR")
                return False
            
            backup_dir = backup_dirs[0]
            
            # Load metadata
            metadata_file = backup_dir / "backup_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                self.log(f"Backup created: {metadata.get('created_at', 'Unknown')}")
            else:
                self.log("⚠️  No metadata found in backup", "WARNING")
            
            # Stop services before restore
            self.log("Stopping services...")
            self.run_command(["docker", "compose", "down"], check=False)
            
            # Restore volumes
            if restore_volumes:
                self.log("Restoring Docker volumes...")
                self._restore_volumes(backup_dir)
            
            # Restore configurations
            if restore_configs:
                self.log("Restoring configurations...")
                self._restore_configs(backup_dir)
            
            # Restore data directories
            if restore_data:
                self.log("Restoring data directories...")
                self._restore_data_dirs(backup_dir)
            
            # Cleanup temporary directory
            shutil.rmtree(restore_dir)
            
            self.log("✅ Backup restore completed successfully!")
            self.log("💡 Run 'python scripts/deploy.py deploy' to restart services")
            
            return True
            
        except Exception as e:
            self.log(f"❌ Backup restore failed: {e}", "ERROR")
            # Cleanup on failure
            if 'restore_dir' in locals() and restore_dir.exists():
                shutil.rmtree(restore_dir)
            return False
    
    def _restore_volumes(self, backup_dir: Path) -> None:
        """Restore Docker volumes"""
        volumes_dir = backup_dir / "volumes"
        if not volumes_dir.exists():
            return
        
        try:
            for volume_file in volumes_dir.glob("*.tar"):
                volume_name = volume_file.stem
                
                # Remove existing volume if it exists
                self.run_command(["docker", "volume", "rm", volume_name], check=False)
                
                # Create new volume
                self.run_command(["docker", "volume", "create", volume_name], check=True)
                
                # Restore volume data
                restore_cmd = [
                    "docker", "run", "--rm", "-v", f"{volume_name}:/data", 
                    "-v", f"{volume_file}:/backup", "alpine", 
                    "tar", "xf", f"/backup", "-C", "/data"
                ]
                
                result = self.run_command(restore_cmd, check=False)
                if result.returncode == 0:
                    self.log(f"✅ Volume {volume_name} restored")
                else:
                    self.log(f"❌ Volume {volume_name} restore failed", "ERROR")
                    
        except Exception as e:
            self.log(f"❌ Volume restore failed: {e}", "ERROR")
    
    def _restore_configs(self, backup_dir: Path) -> None:
        """Restore configuration files"""
        configs_dir = backup_dir / "configs"
        if not configs_dir.exists():
            return
        
        try:
            # Restore individual config files
            for config_file in configs_dir.iterdir():
                if config_file.is_file():
                    target_path = self.project_root / config_file.name
                    shutil.copy2(config_file, target_path)
                    self.log(f"✅ Config file {config_file.name} restored")
                elif config_file.is_dir():
                    target_path = self.project_root / config_file.name
                    if target_path.exists():
                        shutil.rmtree(target_path)
                    shutil.copytree(config_file, target_path)
                    self.log(f"✅ Config directory {config_file.name} restored")
                    
        except Exception as e:
            self.log(f"❌ Config restore failed: {e}", "ERROR")
    
    def _restore_data_dirs(self, backup_dir: Path) -> None:
        """Restore data directories"""
        data_dir = backup_dir / "data"
        if not data_dir.exists():
            return
        
        try:
            for tar_file in data_dir.glob("*.tar"):
                dir_name = tar_file.stem
                target_path = self.project_root / dir_name
                
                # Remove existing directory
                if target_path.exists():
                    shutil.rmtree(target_path)
                
                # Extract data
                with tarfile.open(tar_file, "r") as tar:
                    tar.extractall(self.project_root)
                
                self.log(f"✅ Data directory {dir_name} restored")
                
        except Exception as e:
            self.log(f"❌ Data restore failed: {e}", "ERROR")
    
    def list_backups(self) -> List[Dict]:
        """List available backups"""
        self.log("Available backups:")
        self.log("=" * 50)
        
        backups = []
        
        try:
            for backup_file in self.backup_dir.glob("*.tar.gz"):
                # Get file info
                stat = backup_file.stat()
                size_mb = stat.st_size / (1024 * 1024)
                created = datetime.fromtimestamp(stat.st_mtime)
                
                # Try to load metadata
                metadata_file = self.backup_dir / f"{backup_file.stem}_metadata.json"
                metadata = {}
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                    except:
                        pass
                
                backup_info = {
                    "filename": backup_file.name,
                    "size_mb": size_mb,
                    "created": created,
                    "metadata": metadata
                }
                
                backups.append(backup_info)
                
                # Display backup info
                self.log(f"📦 {backup_file.name}")
                self.log(f"   Size: {size_mb:.2f} MB")
                self.log(f"   Created: {created.strftime('%Y-%m-%d %H:%M:%S')}")
                
                if metadata:
                    components = metadata.get("components", [])
                    completed = sum(1 for c in components if c.get("status") == "completed")
                    total = len(components)
                    self.log(f"   Components: {completed}/{total} completed")
                
                self.log()
            
            if not backups:
                self.log("No backups found")
            
            return backups
            
        except Exception as e:
            self.log(f"❌ Failed to list backups: {e}", "ERROR")
            return []
    
    def cleanup_old_backups(self, keep_days: int = 30, keep_count: int = 10) -> None:
        """Clean up old backups"""
        self.log(f"Cleaning up backups older than {keep_days} days or keeping only {keep_count} most recent")
        
        try:
            # Get all backups
            backups = []
            for backup_file in self.backup_dir.glob("*.tar.gz"):
                stat = backup_file.stat()
                backups.append((backup_file, stat.st_mtime))
            
            # Sort by creation time (newest first)
            backups.sort(key=lambda x: x[1], reverse=True)
            
            # Remove old backups by age
            cutoff_time = time.time() - (keep_days * 24 * 3600)
            removed_by_age = 0
            
            for backup_file, mtime in backups:
                if mtime < cutoff_time:
                    backup_file.unlink()
                    
                    # Remove metadata file if it exists
                    metadata_file = self.backup_dir / f"{backup_file.stem}_metadata.json"
                    if metadata_file.exists():
                        metadata_file.unlink()
                    
                    removed_by_age += 1
                    self.log(f"✅ Removed old backup: {backup_file.name}")
            
            # Remove excess backups by count
            if len(backups) - removed_by_age > keep_count:
                excess_backups = backups[keep_count:]
                removed_by_count = 0
                
                for backup_file, _ in excess_backups:
                    if backup_file.exists():
                        backup_file.unlink()
                        
                        # Remove metadata file if it exists
                        metadata_file = self.backup_dir / f"{backup_file.stem}_metadata.json"
                        if metadata_file.exists():
                            metadata_file.unlink()
                        
                        removed_by_count += 1
                        self.log(f"✅ Removed excess backup: {backup_file.name}")
                
                self.log(f"✅ Cleanup completed: {removed_by_age} old + {removed_by_count} excess backups removed")
            else:
                self.log(f"✅ Cleanup completed: {removed_by_age} old backups removed")
                
        except Exception as e:
            self.log(f"❌ Backup cleanup failed: {e}", "ERROR")
    
    def verify_backup(self, backup_file: str) -> bool:
        """Verify backup integrity"""
        backup_path = Path(backup_file)
        
        if not backup_path.exists():
            self.log(f"❌ Backup file not found: {backup_file}", "ERROR")
            return False
        
        self.log(f"Verifying backup: {backup_file}")
        
        try:
            # Test archive integrity
            with tarfile.open(backup_path, "r:gz") as tar:
                tar.getmembers()  # This will raise an error if archive is corrupted
            
            # Check metadata
            metadata_file = self.backup_dir / f"{backup_path.stem}_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                self.log("✅ Backup verification passed!")
                self.log(f"📊 Components: {len(metadata.get('components', []))}")
                self.log(f"📅 Created: {metadata.get('created_at', 'Unknown')}")
                return True
            else:
                self.log("⚠️  Backup verified but no metadata found", "WARNING")
                return True
                
        except Exception as e:
            self.log(f"❌ Backup verification failed: {e}", "ERROR")
            return False


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="SignaMentis Backup Manager")
    parser.add_argument("action", choices=[
        "create", "restore", "list", "cleanup", "verify"
    ], help="Backup action to perform")
    parser.add_argument("--backup-file", help="Backup file for restore/verify")
    parser.add_argument("--backup-name", help="Custom name for backup")
    parser.add_argument("--backup-dir", default="./backups", help="Backup directory")
    parser.add_argument("--no-volumes", action="store_true", help="Skip volume backup/restore")
    parser.add_argument("--no-configs", action="store_true", help="Skip config backup/restore")
    parser.add_argument("--no-data", action="store_true", help="Skip data backup/restore")
    parser.add_argument("--force", action="store_true", help="Force restore without confirmation")
    parser.add_argument("--keep-days", type=int, default=30, help="Keep backups for N days (cleanup)")
    parser.add_argument("--keep-count", type=int, default=10, help="Keep N most recent backups (cleanup)")
    
    args = parser.parse_args()
    
    backup_manager = BackupManager(args.backup_dir)
    
    try:
        if args.action == "create":
            backup_file = backup_manager.create_backup(
                backup_name=args.backup_name,
                include_volumes=not args.no_volumes,
                include_configs=not args.no_configs,
                include_data=not args.no_data
            )
            
            if backup_file:
                print(f"✅ Backup created: {backup_file}")
                sys.exit(0)
            else:
                print("❌ Backup creation failed")
                sys.exit(1)
        
        elif args.action == "restore":
            if not args.backup_file:
                print("❌ Backup file required for restore")
                sys.exit(1)
            
            success = backup_manager.restore_backup(
                backup_file=args.backup_file,
                restore_volumes=not args.no_volumes,
                restore_configs=not args.no_configs,
                restore_data=not args.no_data,
                force=args.force
            )
            
            sys.exit(0 if success else 1)
        
        elif args.action == "list":
            backup_manager.list_backups()
        
        elif args.action == "cleanup":
            backup_manager.cleanup_old_backups(
                keep_days=args.keep_days,
                keep_count=args.keep_count
            )
        
        elif args.action == "verify":
            if not args.backup_file:
                print("❌ Backup file required for verify")
                sys.exit(1)
            
            success = backup_manager.verify_backup(args.backup_file)
            sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Operation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
