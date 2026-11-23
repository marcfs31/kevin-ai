import { useI18n } from '../i18n';
import { groupTools } from '../lib/toolGroups';
import type { ToolsSent } from '../lib/storage';

/** The tools sent with the last request, grouped by where they come from, each as a chip. The tokens are an estimate. */
export function ToolsViewer({ tools }: { tools: ToolsSent }) {
  const { t, tp, num } = useI18n();
  const groups = groupTools(tools.names);
  return (
    <section className="tools-viewer" aria-label={t('toolsViewer.label')}>
      <h3 className="tools-viewer-title">
        {t('toolsViewer.title')}
        <span className="tools-viewer-total">{t('toolsViewer.total', { count: tp('tools.count', tools.names.length), tokens: num(tools.tokens) })}</span>
      </h3>
      {groups.map((g) => (
        <div key={g.id} className="tool-group">
          <h4>
            {g.builtin ? t('toolsViewer.builtin') : g.id}
            <span className="tool-group-count">{g.tools.length}</span>
          </h4>
          <ul className="tool-chips">
            {g.tools.map((x) => (
              <li key={x.name} title={x.name}>
                <code>{x.label}</code>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </section>
  );
}
