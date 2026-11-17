import { useEffect, useMemo, useState } from "react";
import { getDocumentMindmap, getErrorMessage } from "../../services/api";

const W = 720;
const H = 460;

// Dependency-free layout: the most connected nodes sit near the centre, the rest on a ring.
function layout(nodes, edges) {
  const degree = new Map(nodes.map((n) => [n.id, 0]));
  edges.forEach((e) => {
    degree.set(e.source, (degree.get(e.source) || 0) + 1);
    degree.set(e.target, (degree.get(e.target) || 0) + 1);
  });
  const sorted = [...nodes].sort((a, b) => (degree.get(b.id) || 0) - (degree.get(a.id) || 0));
  const pos = new Map();
  const hubs = sorted.slice(0, Math.min(3, sorted.length));
  hubs.forEach((n, i) => {
    const a = (2 * Math.PI * i) / Math.max(hubs.length, 1);
    const r = hubs.length === 1 ? 0 : 60;
    pos.set(n.id, { x: W / 2 + r * Math.cos(a), y: H / 2 + r * Math.sin(a) });
  });
  const rest = sorted.slice(hubs.length);
  rest.forEach((n, i) => {
    const ring = i < 14 ? 0 : 1;
    const inRing = ring === 0 ? Math.min(rest.length, 14) : rest.length - 14;
    const idx = ring === 0 ? i : i - 14;
    const a = (2 * Math.PI * idx) / Math.max(inRing, 1) + ring * 0.2;
    const rx = ring === 0 ? W * 0.34 : W * 0.46;
    const ry = ring === 0 ? H * 0.34 : H * 0.46;
    pos.set(n.id, { x: W / 2 + rx * Math.cos(a), y: H / 2 + ry * Math.sin(a) });
  });
  return { pos, degree };
}

export default function MindmapPanel({ documentId, onClose }) {
  const [state, setState] = useState({ loading: true, error: null, data: null });
  const [selected, setSelected] = useState(null);

  useEffect(() => {
    let cancelled = false;
    getDocumentMindmap(documentId)
      .then((res) => !cancelled && setState({ loading: false, error: null, data: res.data }))
      .catch((err) =>
        !cancelled &&
        setState({ loading: false, error: getErrorMessage(err, "Không tải được sơ đồ tư duy."), data: null })
      );
    return () => {
      cancelled = true;
    };
  }, [documentId]);

  const { nodes, edges } = state.data || { nodes: [], edges: [] };
  const { pos, degree } = useMemo(() => layout(nodes, edges), [nodes, edges]);
  const byId = useMemo(() => new Map(nodes.map((n) => [n.id, n])), [nodes]);
  const related = useMemo(() => {
    if (!selected) return null;
    const ids = new Set([selected]);
    edges.forEach((e) => {
      if (e.source === selected) ids.add(e.target);
      if (e.target === selected) ids.add(e.source);
    });
    return ids;
  }, [selected, edges]);

  return (
    <div className="mindmap" role="region" aria-label="Sơ đồ tư duy">
      <div className="mindmap-head">
        <b>Sơ đồ tư duy</b>
        {state.data?.enabled && <span className="mindmap-stats">{nodes.length} khái niệm · {edges.length} quan hệ</span>}
        <button type="button" className="mindmap-close" onClick={onClose} aria-label="Đóng sơ đồ tư duy">×</button>
      </div>
      {state.loading && <p className="msg-extra">Đang tải…</p>}
      {state.error && <p className="note note--err">{state.error}</p>}
      {state.data && !state.data.enabled && (
        <p className="msg-extra">Đồ thị tri thức đang tắt (đặt KG_ENABLED=true ở backend rồi tải lại tài liệu).</p>
      )}
      {state.data?.enabled && nodes.length === 0 && (
        <p className="msg-extra">Chưa có khái niệm nào cho tài liệu này.</p>
      )}
      {nodes.length > 0 && (
        <svg viewBox={`0 0 ${W} ${H}`} className="mindmap-svg">
          {edges.map((e) => {
            const a = pos.get(e.source);
            const b = pos.get(e.target);
            if (!a || !b) return null;
            const dim = related && !(related.has(e.source) && related.has(e.target));
            return (
              <g key={e.id} opacity={dim ? 0.15 : 0.8}>
                <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} className="mindmap-edge" />
                {(!related || !dim) && (
                  <text x={(a.x + b.x) / 2} y={(a.y + b.y) / 2} className="mindmap-rel">{e.relation}</text>
                )}
              </g>
            );
          })}
          {nodes.map((n) => {
            const p = pos.get(n.id);
            const dim = related && !related.has(n.id);
            const r = 5 + Math.min(8, degree.get(n.id) || 0);
            return (
              <g
                key={n.id}
                opacity={dim ? 0.25 : 1}
                onClick={() => setSelected(selected === n.id ? null : n.id)}
                style={{ cursor: "pointer" }}
              >
                <circle cx={p.x} cy={p.y} r={r} className={`mindmap-node${selected === n.id ? " is-sel" : ""}`} />
                <text x={p.x} y={p.y - r - 4} textAnchor="middle" className="mindmap-label">{n.name}</text>
              </g>
            );
          })}
        </svg>
      )}
      {selected && byId.get(selected) && (
        <p className="msg-extra">
          <b>{byId.get(selected).name}</b>
          {byId.get(selected).chunk_indices?.length > 0 &&
            ` — xuất hiện ở đoạn ${byId.get(selected).chunk_indices.slice(0, 8).join(", ")}`}
        </p>
      )}
    </div>
  );
}
