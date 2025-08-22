#!/usr/bin/env python3
"""
GitHub Repository Setup Script for SignaMentis

This script helps set up a new GitHub repository for the SignaMentis project.
It provides instructions and commands to create the repository manually.

Author: SignaMentis Team
Version: 1.0.0
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header():
    """Print script header."""
    print("🚀 SignaMentis GitHub Repository Setup")
    print("=" * 50)
    print()

def check_git_status():
    """Check current git status."""
    print("📋 Checking Git Status...")
    
    try:
        # Check if we're in a git repository
        result = subprocess.run(['git', 'status'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Git repository initialized")
            
            # Check current branch
            branch_result = subprocess.run(['git', 'branch', '--show-current'], 
                                        capture_output=True, text=True)
            if branch_result.returncode == 0:
                print(f"📍 Current branch: {branch_result.stdout.strip()}")
            
            # Check remote
            remote_result = subprocess.run(['git', 'remote', '-v'], 
                                        capture_output=True, text=True)
            if remote_result.returncode == 0 and remote_result.stdout.strip():
                print("🔗 Remote configured:")
                print(remote_result.stdout.strip())
            else:
                print("⚠️  No remote configured")
                
        else:
            print("❌ Not in a git repository")
            return False
            
    except Exception as e:
        print(f"❌ Error checking git status: {e}")
        return False
    
    return True

def create_github_repo_instructions():
    """Provide instructions for creating GitHub repository."""
    print("\n📝 GitHub Repository Setup Instructions:")
    print("-" * 40)
    
    print("1. Go to https://github.com/new")
    print("2. Repository name: signamentis")
    print("3. Description: AI-Powered Trading System for XAU/USD")
    print("4. Make it Public (recommended for open source)")
    print("5. Add .gitignore: Python")
    print("6. Add README.md: Yes")
    print("7. Add license: MIT License")
    print("8. Click 'Create repository'")
    print()
    
    print("After creating the repository, you'll see setup instructions.")
    print("Use the 'push an existing repository' commands:")
    print()

def show_push_commands():
    """Show the commands to push to GitHub."""
    print("🚀 Push Commands (run these after creating the repository):")
    print("-" * 40)
    
    print("git remote add origin https://github.com/YOUR_USERNAME/signamentis.git")
    print("git branch -M main")
    print("git push -u origin main")
    print()
    
    print("Replace YOUR_USERNAME with your actual GitHub username!")
    print()

def setup_github_actions():
    """Set up GitHub Actions workflow."""
    print("⚙️  Setting up GitHub Actions...")
    
    # Create .github/workflows directory
    workflows_dir = Path(".github/workflows")
    workflows_dir.mkdir(parents=True, exist_ok=True)
    
    # Create CI/CD workflow
    workflow_file = workflows_dir / "ci.yml"
    
    workflow_content = """name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        python-version: [3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        python -m pytest tests/ -v
    
    - name: Run deployment tests
      run: |
        python scripts/test_deployment.py
    
    - name: Check code quality
      run: |
        pip install flake8 black isort
        flake8 scripts/ --max-line-length=120
        black --check scripts/
        isort --check-only scripts/

  docker:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Build Docker images
      run: |
        docker build -t signamentis:latest .
        docker build -t signamentis/news-nlp:latest services/news_nlp/
    
    - name: Test Docker Compose
      run: |
        docker-compose config
        docker-compose up -d
        sleep 30
        docker-compose down
"""
    
    with open(workflow_file, 'w') as f:
        f.write(workflow_content)
    
    print(f"✅ Created GitHub Actions workflow: {workflow_file}")

def create_github_issue_templates():
    """Create GitHub issue templates."""
    print("📋 Creating GitHub Issue Templates...")
    
    # Create .github/ISSUE_TEMPLATE directory
    issue_dir = Path(".github/ISSUE_TEMPLATE")
    issue_dir.mkdir(parents=True, exist_ok=True)
    
    # Bug report template
    bug_template = """---
name: Bug report
about: Create a report to help us improve
title: '[BUG] '
labels: ['bug']
assignees: ''
---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment:**
 - OS: [e.g. Ubuntu 20.04]
 - Python Version: [e.g. 3.10.6]
 - SignaMentis Version: [e.g. 1.0.0]

**Additional context**
Add any other context about the problem here.
"""
    
    bug_file = issue_dir / "bug_report.md"
    with open(bug_file, 'w') as f:
        f.write(bug_template)
    
    # Feature request template
    feature_template = """---
name: Feature request
about: Suggest an idea for this project
title: '[FEATURE] '
labels: ['enhancement']
assignees: ''
---

**Is your feature request related to a problem? Please describe.**
A clear and concise description of what the problem is. Ex. I'm always frustrated when [...]

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
A clear and concise description of any alternative solutions or features you've considered.

**Additional context**
Add any other context or screenshots about the feature request here.
"""
    
    feature_file = issue_dir / "feature_request.md"
    with open(feature_file, 'w') as f:
        f.write(feature_template)
    
    print(f"✅ Created issue templates in {issue_dir}")

def main():
    """Main function."""
    print_header()
    
    # Check git status
    if not check_git_status():
        print("❌ Please initialize git repository first:")
        print("   git init")
        print("   git add .")
        print("   git commit -m 'Initial commit'")
        return
    
    print()
    
    # Provide setup instructions
    create_github_repo_instructions()
    
    # Show push commands
    show_push_commands()
    
    # Set up GitHub Actions
    setup_github_actions()
    
    # Create issue templates
    create_github_issue_templates()
    
    print("\n🎉 GitHub setup complete!")
    print("\nNext steps:")
    print("1. Create the repository on GitHub.com")
    print("2. Run the push commands shown above")
    print("3. Check GitHub Actions tab for CI/CD pipeline")
    print("4. Start collaborating! 🚀")

if __name__ == "__main__":
    main()

