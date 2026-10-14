import { pageOf } from "src/lib/tables";

export const PAGE_SIZE = 10;

/** "1–10 of 344" with previous and next buttons; hidden when everything fits on one page. */
export function Pager({ total, page, onPage }: { total: number; page: number; onPage: (p: number) => void }) {
  const p = pageOf(total, page, PAGE_SIZE);
  if (p.pages <= 1) return null;
  return (
    <div className="pager" role="navigation" aria-label="Table pages">
      <button className="pager-btn" aria-label="Previous page" disabled={p.page === 0} onClick={() => onPage(p.page - 1)}>‹</button>
      <span className="num">
        {p.start + 1}–{p.end} of {total}
      </span>
      <button className="pager-btn" aria-label="Next page" disabled={p.page === p.pages - 1} onClick={() => onPage(p.page + 1)}>›</button>
    </div>
  );
}
