export type ToolResult = { content: Array<{ type: "text"; text: string }>; isError?: boolean };

export function text(s: string): ToolResult {
  return { content: [{ type: "text" as const, text: s }] };
}

export function errorResult(e: unknown): ToolResult {
  const msg = e instanceof Error ? e.message : String(e);
  return { content: [{ type: "text" as const, text: `Error: ${msg}` }], isError: true };
}

/** Wrap a tool handler with error catching that returns MCP-friendly error text instead of throwing.
 *  Preserves the MCP SDK `extra` parameter for access to sendNotification, signal, etc. */
export function safeTool<T extends Record<string, unknown>>(
  handler: (args: T, extra?: any) => Promise<ToolResult>
): (args: T, extra?: any) => Promise<ToolResult> {
  return async (args: T, extra?: any) => {
    try {
      return await handler(args, extra);
    } catch (e) {
      return errorResult(e);
    }
  };
}
