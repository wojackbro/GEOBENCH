#!/bin/bash
# Push this repo to GitHub. Run from the project root.
#
# 1. Create a new repo on GitHub: https://github.com/new
#    Name it e.g. geo_ai1 (or anything you like).
# 2. Set your GitHub username and run:
#
#    export GITHUB_USERNAME=your-github-username
#    ./push_to_github.sh
#
# Or in one line:
#    GITHUB_USERNAME=your-github-username ./push_to_github.sh
#
# If you use GitHub CLI and are logged in, you can instead run:
#    gh repo create geo_ai1 --private --source=. --remote=origin --push

set -e
cd "$(dirname "$0")"

if [ -z "$GITHUB_USERNAME" ]; then
  echo "Set your GitHub username:"
  echo "  export GITHUB_USERNAME=your-username"
  echo "  ./push_to_github.sh"
  echo ""
  echo "Or create and push in one go (if you have 'gh' and are logged in):"
  echo "  gh repo create geo_ai1 --private --source=. --remote=origin --push"
  exit 1
fi

REPO_NAME="${GITHUB_REPO_NAME:-geo_ai1}"
URL="https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"

if git remote get-url origin 2>/dev/null | grep -q REPLACE_WITH_YOUR_USERNAME; then
  git remote set-url origin "$URL"
  echo "Set origin to $URL"
fi

git push -u origin main
echo "Pushed to $URL"
