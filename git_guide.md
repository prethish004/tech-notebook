# Git & GitHub Complete Guide

## Table of Contents
1. [What is Git?](#what-is-git)
2. [Installation & Setup](#installation--setup)
3. [Basic Commands](#basic-commands)
4. [Repository Operations](#repository-operations)
5. [Cloning Repositories](#cloning-repositories)
6. [Branching & Merging](#branching--merging)
7. [Committing Changes](#committing-changes)
8. [Remote Operations](#remote-operations)
9. [Undoing Changes](#undoing-changes)
10. [GitHub License Guide](#github-license-guide)
11. [Common Workflows](#common-workflows)

---

## What is Git?

Git is a distributed version control system that allows you to:
- Track changes to your code over time
- Collaborate with other developers
- Maintain multiple versions of your project
- Revert to previous versions if needed

**Key Concepts:**
- **Repository (Repo):** A folder containing your project and its version history
- **Commit:** A snapshot of your code at a specific point in time
- **Branch:** A parallel version of your repository
- **Remote:** A version of your repository hosted online (e.g., GitHub)
- **Local:** The repository on your computer

---

## Installation & Setup

### On Windows
```bash
# Download and install from
https://git-scm.com/download/win

# Or use Chocolatey
choco install git
```

### On macOS
```bash
# Using Homebrew
brew install git

# Or download from
https://git-scm.com/download/mac
```

### On Linux
```bash
# Ubuntu/Debian
sudo apt-get install git

# Fedora/CentOS
sudo yum install git
```

### Configure Git
```bash
# Set your username
git config --global user.name "Your Name"

# Set your email
git config --global user.email "your.email@example.com"

# Verify configuration
git config --list

# Configure default branch name (optional, modern default is 'main')
git config --global init.defaultBranch main
```

---

## Basic Commands

### Initialize a Repository
```bash
# Create a new Git repository in current directory
git init

# Initialize with specific directory
git init [directory-name]
```

### Check Repository Status
```bash
# Show status of working directory and staging area
git status

# Short format
git status -s

# Show branch information only
git status -b
```

### View Commit History
```bash
# Show all commits
git log

# Show last N commits
git log -n 5

# Show commits in one line
git log --oneline

# Show commits with visual branch graph
git log --graph --oneline --all

# Show commits from specific author
git log --author="Name"

# Show commits from last 7 days
git log --since="7 days ago"
```

### View Changes
```bash
# Show unstaged changes
git diff

# Show staged changes
git diff --staged

# Show changes in specific file
git diff [filename]

# Compare between branches
git diff branch1 branch2

# Show changes in last commit
git diff HEAD~1 HEAD
```

---

## Repository Operations

### Clone a Repository

#### Clone Public Repository
```bash
# Clone via HTTPS (no authentication needed)
git clone https://github.com/username/repository-name.git

# Clone and rename the directory
git clone https://github.com/username/repository-name.git my-folder-name

# Clone specific branch
git clone --branch branch-name https://github.com/username/repository-name.git

# Clone with depth (faster for large repos)
git clone --depth 1 https://github.com/username/repository-name.git
```

**When to use HTTPS:**
- Public repositories
- No SSH key setup required
- Works behind firewalls

#### Clone Private Repository

**Option 1: Using HTTPS with Personal Access Token (Recommended)**
```bash
# Generate Personal Access Token on GitHub:
# Settings > Developer Settings > Personal Access Tokens > Generate new token
# Select scopes: repo (for full control of private repositories)

# Clone using token
git clone https://[TOKEN]@github.com/username/private-repo.git

# Or you'll be prompted for username and password
git clone https://github.com/username/private-repo.git
# When prompted for password, paste your Personal Access Token
```

**Option 2: Using SSH (More Secure)**
```bash
# First, generate SSH key (if you don't have one)
ssh-keygen -t ed25519 -C "your.email@example.com"

# Or for older systems
ssh-keygen -t rsa -b 4096 -C "your.email@example.com"

# Add key to SSH agent
eval $(ssh-agent -s)
ssh-add ~/.ssh/id_ed25519

# Add public key to GitHub:
# Settings > SSH and GPG keys > New SSH key
# Paste contents of ~/.ssh/id_ed25519.pub

# Clone using SSH
git clone git@github.com:username/private-repo.git

# Test SSH connection
ssh -T git@github.com
```

**Option 3: Using Git Credential Manager**
```bash
# Install Git Credential Manager
# Windows: Comes with Git for Windows
# macOS: brew install --cask git-credential-manager
# Linux: Download from https://github.com/git-ecosystem/git-credential-manager

# Configure to use credential manager
git config --global credential.helper manager

# Clone normally (it will prompt for authentication once)
git clone https://github.com/username/private-repo.git
```

---

## Branching & Merging

### Create & Switch Branches
```bash
# Create a new branch
git branch branch-name

# Create and switch to new branch (one command)
git checkout -b branch-name

# Or modern syntax
git switch -c branch-name

# List all branches (local)
git branch

# List all branches (including remote)
git branch -a

# List branches with latest commit info
git branch -v
```

### Switch Branches
```bash
# Switch to existing branch
git checkout branch-name

# Or modern syntax
git switch branch-name

# Undo last branch switch
git checkout -
```

### Delete Branches
```bash
# Delete local branch
git branch -d branch-name

# Force delete (even if not merged)
git branch -D branch-name

# Delete remote branch
git push origin --delete branch-name
```

### Merge Branches
```bash
# Merge a branch into current branch
git merge branch-name

# Merge with commit message
git merge branch-name -m "Merge message"

# Merge with no fast-forward (creates merge commit)
git merge --no-ff branch-name

# Abort a merge (if conflicts)
git merge --abort
```

---

## Committing Changes

### Stage Changes
```bash
# Stage specific file
git add [filename]

# Stage all changes
git add .

# Stage all changes in current directory
git add -A

# Stage interactively (choose specific changes)
git add -p

# Unstage file
git reset [filename]

# Unstage all changes
git reset
```

### Create Commits
```bash
# Commit staged changes
git commit -m "Commit message"

# Commit with detailed message
git commit -m "Short summary" -m "Detailed description"

# Stage and commit (tracked files only)
git commit -am "Message"

# Amend last commit (don't push if already pushed!)
git commit --amend

# Amend with new message
git commit --amend -m "New message"

# Create empty commit
git commit --allow-empty -m "Message"
```

**Commit Message Best Practices:**
```
feat: Add user authentication
fix: Resolve login button crash
docs: Update API documentation
style: Format code with prettier
refactor: Simplify database queries
test: Add unit tests for auth
chore: Update dependencies
```

---

## Remote Operations

### Add & Manage Remotes
```bash
# Show all remotes
git remote

# Show remotes with URLs
git remote -v

# Add a remote
git remote add origin https://github.com/username/repo.git

# Change remote URL
git remote set-url origin https://github.com/username/new-repo.git

# Remove remote
git remote remove origin

# Show remote details
git remote show origin
```

### Push Changes
```bash
# Push current branch to remote
git push

# Push to specific remote and branch
git push origin main

# Push all branches
git push origin --all

# Push tags
git push origin --tags

# Push specific tag
git push origin tag-name

# Force push (use with caution!)
git push --force

# Safe force push
git push --force-with-lease
```

### Pull Changes
```bash
# Fetch and merge remote changes
git pull

# Pull from specific remote and branch
git pull origin main

# Pull with rebase instead of merge
git pull --rebase

# Only fetch without merging
git fetch

# Fetch all branches
git fetch --all
```

---

## Undoing Changes

### Undo Uncommitted Changes
```bash
# Discard changes in working directory
git restore [filename]

# Or older syntax
git checkout -- [filename]

# Discard all changes
git restore .

# Unstage changes
git restore --staged [filename]
```

### Undo Commits
```bash
# Undo last commit (keep changes staged)
git reset --soft HEAD~1

# Undo last commit (keep changes unstaged)
git reset --mixed HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1

# Create new commit that undoes changes
git revert HEAD

# Go to specific commit (detached HEAD)
git checkout commit-hash

# Go back to previous state
git reflog
git reset --hard HEAD@{n}
```

### Fix Mistakes
```bash
# See all your actions
git reflog

# Recover deleted branch
git reflog
git checkout -b recovered-branch commit-hash

# Find lost commits
git fsck --lost-found
```

---

## GitHub License Guide

### What is a License?
A license tells others what they can and cannot do with your code.

### Common Licenses

#### MIT License (Recommended for most projects)
- **Usage:** Permissive, allows commercial use
- **Requirements:** Include license and copyright notice
- **Restrictions:** No liability, no warranty
- **Best for:** Open-source projects, learning projects
```
Add LICENSE file with MIT text:
https://opensource.org/licenses/MIT
```

#### Apache 2.0 License
- **Usage:** Permissive, allows commercial use
- **Requirements:** Include license, list changes
- **Restrictions:** No liability, trademark rights
- **Best for:** Professional open-source projects

#### GPL v3 License (Copyleft)
- **Usage:** Code must remain open-source
- **Requirements:** Include source code, disclose changes
- **Restrictions:** Any derivative must use same license
- **Best for:** When you want to keep code free

#### BSD License
- **Usage:** Similar to MIT, very permissive
- **Requirements:** Include license and copyright
- **Restrictions:** No liability, no warranty
- **Best for:** Academic and commercial projects

#### ISC License
- **Usage:** Very simple and permissive
- **Requirements:** Include license
- **Restrictions:** No liability
- **Best for:** Simple projects

### Add License to GitHub

**Method 1: Using GitHub Web Interface**
1. Go to your repository
2. Click "Add file" > "Create new file"
3. Name it `LICENSE`
4. GitHub will suggest licenses - select one
5. Review and commit

**Method 2: Using Terminal**
```bash
# Create LICENSE file
touch LICENSE

# Add license text (example: MIT)
# Copy MIT license from: https://opensource.org/licenses/MIT

git add LICENSE
git commit -m "Add MIT license"
git push origin main
```

**Method 3: Initialize with License**
```bash
# When creating a new GitHub repo, select a license in the dropdown
```

---

## Common Workflows

### Feature Branch Workflow
```bash
# 1. Create feature branch from main
git checkout -b feature/user-authentication

# 2. Make changes and commit
git add .
git commit -m "Implement user login"

# 3. Push to remote
git push origin feature/user-authentication

# 4. Create Pull Request on GitHub

# 5. After merge, delete branch
git branch -d feature/user-authentication
```

### Daily Development Workflow
```bash
# 1. Start day - update main branch
git checkout main
git pull origin main

# 2. Create feature branch
git checkout -b feature/new-feature

# 3. Make changes
git add .
git commit -m "Work on feature"

# 4. Push regularly
git push origin feature/new-feature

# 5. End of day - push all changes
git push origin feature/new-feature
```

### Fixing Mistakes Workflow
```bash
# 1. Realize you made a mistake
git log --oneline

# 2. Option A: Amend last commit (not pushed)
git commit --amend -m "Corrected message"

# 3. Option B: Revert commit (already pushed)
git revert commit-hash
git push origin main

# 4. Option C: Reset to previous state
git reset --hard HEAD~1
```

### Syncing Fork with Upstream
```bash
# 1. Add upstream remote
git remote add upstream https://github.com/original-owner/repo.git

# 2. Fetch latest changes from upstream
git fetch upstream

# 3. Switch to main branch
git checkout main

# 4. Merge upstream changes
git merge upstream/main

# 5. Push to your fork
git push origin main
```

---

## Tips & Tricks

### Useful Aliases
```bash
# Add to ~/.gitconfig or use git config

git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.unstage 'restore --staged'
git config --global alias.last 'log -1 HEAD'
git config --global alias.visual 'log --graph --oneline --all'

# Usage
git st    # Instead of git status
git co    # Instead of git checkout
git br    # Instead of git branch
```

### .gitignore Template
```
# Python
__pycache__/
*.py[cod]
*$py.class
venv/
.env

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Build
dist/
build/
*.egg-info/
```

### Emergency Commands
```bash
# See what would be deleted
git clean -n

# Remove untracked files
git clean -fd

# Stash changes temporarily
git stash

# Retrieve stashed changes
git stash pop

# List stashes
git stash list
```

---

## Troubleshooting

### Large Files / Slow Clone
```bash
# Clone with depth limit (faster)
git clone --depth 1 https://github.com/username/repo.git

# Later, fetch full history
git fetch --unshallow
```

### Authentication Issues
```bash
# Clear stored credentials and re-authenticate
git config --global --unset credential.helper
git config --global credential.helper manager

# Or use SSH instead
git remote set-url origin git@github.com:username/repo.git
```

### Merge Conflicts
```bash
# See conflicted files
git status

# Edit files to resolve conflicts (marked with <<<<<<, ======, >>>>>>)

# After fixing
git add .
git commit -m "Resolve merge conflicts"
git push
```

### Accidentally Pushed to Wrong Branch
```bash
# Revert the commit on remote
git revert HEAD
git push

# Or reset (if you have permission and team agrees)
git reset --hard HEAD~1
git push --force-with-lease
```

---

## Resources

- **Official Git Documentation:** https://git-scm.com/doc
- **GitHub Guides:** https://guides.github.com
- **Atlassian Git Tutorials:** https://www.atlassian.com/git
- **Interactive Learning:** https://learngitbranching.js.org
- **Git Cheat Sheet:** https://github.github.com/training-kit/

---

**Last Updated:** January 2026
**Difficulty Level:** Beginner to Intermediate