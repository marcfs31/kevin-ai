import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid(
  defineConfig({
    title: 'Kevin AI',
    description: 'Private local AI assistant — architecture, APIs, and how models run',
    base: '/kevin-ai/',
    lastUpdated: true,

    themeConfig: {
      nav: [
        { text: 'Home', link: '/' },
        { text: 'Guide', link: '/guide/overview' },
        {
          text: 'GitDiagram',
          link: 'https://gitdiagram.com/marcfs31/kevin-ai',
        },
        {
          text: 'GitHub',
          link: 'https://github.com/marcfs31/kevin-ai',
        },
      ],

      sidebar: [
        {
          text: 'Introduction',
          items: [{ text: 'What Kevin AI is', link: '/' }],
        },
        {
          text: 'Architecture',
          items: [
            { text: 'System overview', link: '/guide/overview' },
            { text: 'Vite vs chat_server', link: '/guide/vite-and-chat-server' },
            { text: 'How the APIs are built', link: '/guide/apis' },
            { text: 'How models run', link: '/guide/models' },
            { text: 'Run the demo on the Mac', link: '/guide/run-on-mac' },
            { text: 'Repository map (GitDiagram)', link: '/guide/gitdiagram' },
          ],
        },
      ],

      socialLinks: [
        { icon: 'github', link: 'https://github.com/marcfs31/kevin-ai' },
      ],

      outline: { level: [2, 3] },
      search: { provider: 'local' },
    },
  }),
)
