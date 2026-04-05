import { describe, it, expect, vi } from "vitest";
import { safeTool, text, errorResult } from "../tool-helpers.js";

describe("safeTool", () => {
  it("passes extra parameter to handler", async () => {
    const handler = vi.fn().mockResolvedValue(text("ok"));
    const wrapped = safeTool(handler);

    const fakeExtra = { sendNotification: vi.fn(), signal: new AbortController().signal };
    await wrapped({ foo: "bar" }, fakeExtra);

    expect(handler).toHaveBeenCalledWith({ foo: "bar" }, fakeExtra);
  });

  it("catches errors and returns MCP-friendly error with extra present", async () => {
    const handler = vi.fn().mockRejectedValue(new Error("boom"));
    const wrapped = safeTool(handler);

    const fakeExtra = { sendNotification: vi.fn() };
    const result = await wrapped({ x: 1 }, fakeExtra);

    expect(result.isError).toBe(true);
    expect(result.content[0].text).toBe("Error: boom");
  });

  it("works without extra parameter (backward compat)", async () => {
    const handler = vi.fn().mockResolvedValue(text("done"));
    const wrapped = safeTool(handler);

    const result = await wrapped({ a: "b" });

    expect(result.content[0].text).toBe("done");
    expect(handler).toHaveBeenCalledWith({ a: "b" }, undefined);
  });
});

describe("text helper", () => {
  it("wraps string in MCP tool result format", () => {
    const result = text("hello");
    expect(result).toEqual({
      content: [{ type: "text", text: "hello" }],
    });
  });
});

describe("errorResult helper", () => {
  it("wraps Error in MCP error result", () => {
    const result = errorResult(new Error("fail"));
    expect(result.isError).toBe(true);
    expect(result.content[0].text).toBe("Error: fail");
  });

  it("wraps non-Error in MCP error result", () => {
    const result = errorResult("string error");
    expect(result.content[0].text).toBe("Error: string error");
  });
});
