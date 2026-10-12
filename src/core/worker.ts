import { loadCore } from "src/core/loader";
import { createHandler, type Request, type Response } from "src/core/protocol";
import { createSession } from "src/core/session";

const scope = self as unknown as {
  onmessage: ((e: MessageEvent<Request>) => void) | null;
  postMessage(msg: Response): void;
};

const handle = createHandler((opts, onProgress) => loadCore(opts, createSession, onProgress));
scope.onmessage = (e) => void handle(e.data, (r) => scope.postMessage(r));
