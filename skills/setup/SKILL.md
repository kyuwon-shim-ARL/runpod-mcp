---
name: setup
description: Configure RunPod API key and SSH key for the runpod-mcp plugin
---

# RunPod MCP Setup

Guide the user through configuring their RunPod API credentials.

## Steps

1. Ask the user if they already have a RunPod API key. If not, direct them to:
   https://www.runpod.io/console/user/settings

2. Once they have the key, instruct them to add these lines to their shell profile (`~/.bashrc` or `~/.zshrc`):

```bash
export RUNPOD_API_KEY=rp_xxxxxx
```

3. (Optional) If they want SSH/rsync features, ask for their SSH key path:

```bash
export SSH_KEY_PATH=~/.ssh/id_ed25519
```

4. Remind them to restart Claude Code (or open a new terminal) for the changes to take effect.

5. Verify the setup by calling the `list_pods` tool. If it returns without error, the setup is complete.

## Troubleshooting

- If tools return "RUNPOD_API_KEY is not configured", the environment variable is not reaching the MCP server. Check that it's exported (not just set) and that Claude Code was restarted.
- If SSH commands fail, verify the SSH key path exists and has correct permissions (chmod 600).
- API keys start with `rp_`. If the key looks different, it may be invalid.
