#!/bin/bash
set -e

# Handle different commands
case "$1" in
  shell)
    exec /bin/bash
    ;;
  chat)
    # Run the chat interface
    if [ -f "chat.py" ]; then
      exec python chat.py
    else
      echo "No chat.py found in workspace"
      exit 1
    fi
    ;;
  train)
    # Run training
    if [ -f "train.py" ]; then
      exec python train.py "${@:2}"
    else
      echo "No train.py found in workspace"
      exit 1
    fi
    ;;
  *)
    # Default: run create-llm CLI
    exec node /usr/local/lib/create-llm/dist/index.js "$@"
    ;;
esac
