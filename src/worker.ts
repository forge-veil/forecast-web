import { loadCore } from "src/loader";
import { createHandler, type Request, type Response } from "src/protocol";
import { createSession } from "src/session";

const scope = self as unknown as {
  onmessage: ((e: MessageEvent<Request>) => void) | null;
  postMessage(msg: Response): void;
};

const handle = createHandler((opts, onProgress) => loadCore(opts, createSession, onProgress));
scope.onmessage = (e) => void handle(e.data, (r) => scope.postMessage(r));
