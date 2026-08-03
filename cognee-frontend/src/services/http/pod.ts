import { createHttpClient } from "./client";
import { reportClientLog } from "./reportClientLog";

function inputPath(input: RequestInfo | URL): string {
  if (typeof input === "string") {
    if (input.startsWith("/")) return input;
    const url = new URL(input);
    return `${url.pathname}${url.search}`;
  }

  const url = input instanceof Request ? new URL(input.url) : input;
  return `${url.pathname}${url.search}`;
}

function headersToRecord(headers: HeadersInit | undefined): Record<string, string> {
  if (!headers) return {};
  if (headers instanceof Headers) return Object.fromEntries(headers.entries());
  if (Array.isArray(headers)) return Object.fromEntries(headers);
  return headers;
}

/**
 * Build an HTTP client for a tenant pod.
 *
 * Pod requests are made directly by the browser, so this wrapper anchors every
 * request to the configured service URL, injects the non-overridable API key,
 * and forwards failures to the same-origin client-log endpoint.
 */
export function createPodClient(serviceUrl: string, apiKey: string) {
  const client = createHttpClient();
  const root = serviceUrl.replace(/\/$/, "");

  client.setLogger((event) => {
    if (!event.error) return;
    const message = event.error instanceof Error ? event.error.message : String(event.error);
    reportClientLog("POD-API", "error", message, {
      url: event.url,
      method: event.method,
    });
  });

  return {
    ...client,
    fetch(input: RequestInfo | URL, init: RequestInit = {}): Promise<Response> {
      const path = inputPath(input);
      return client.request(`${root}/api${path}`, {
        ...init,
        credentials: "omit",
        headers: {
          ...headersToRecord(init.headers),
          "X-Api-Key": apiKey,
        },
      });
    },
  };
}
