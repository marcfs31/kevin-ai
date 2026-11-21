import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { App } from './App';
import { BACKUP_KEY, STORAGE_KEY } from './lib/storage';
import { DONE, FINISH, FINISH_TOOLS, USAGE, axeViolations, delta, mockFetch, modelsResponse, sseResponse, toolCallDelta } from './test/helpers';

function renderApp(languages: readonly string[] = ['en-US']) {
  return render(<App storage={localStorage} languages={languages} />);
}

function typeAndSend(text: string) {
  const box = screen.getByLabelText('Message');
  fireEvent.change(box, { target: { value: text } });
  fireEvent.keyDown(box, { key: 'Enter' });
}

function bodyOf(fetchMock: ReturnType<typeof mockFetch>, call: number) {
  const chats = fetchMock.mock.calls.filter(([url]) => String(url).endsWith('/v1/chat/completions'));
  return JSON.parse(String(chats[call]?.[1]?.body));
}

/** A reply that never ends until the request is aborted. */
function heldReply(init: RequestInit | undefined) {
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(delta('partial')));
      init?.signal?.addEventListener('abort', () => controller.error(new DOMException('aborted', 'AbortError')));
    },
  });
  return new Response(stream, { status: 200 });
}

/** The number in the box with this label inside a list of boxes. */
const boxValue = (list: HTMLElement, label: string) => within(list).getByText(label).nextElementSibling?.textContent;

describe('App', () => {
  it('sends with Enter, streams the reply and shows its statistics', async () => {
    const fetchMock = mockFetch(() => sseResponse([delta('Hello'), delta(' **world**'), FINISH, USAGE, DONE]));
    renderApp();
    await screen.findByText('Model: qwen3.5-4b');
    typeAndSend('hi there');
    expect(await screen.findByText('world')).toBeTruthy();
    expect(within(screen.getByRole('log')).getByText('hi there')).toBeTruthy();
    expect(within(screen.getByRole('log')).getByText('You')).toBeTruthy();
    expect(within(screen.getByRole('log')).getByText('Assistant')).toBeTruthy();
    const stats = await screen.findByLabelText('Reply statistics');
    expect(stats.textContent).toContain('12 prompt tokens');
    expect(stats.textContent).toContain('8 from cache');
    expect(bodyOf(fetchMock, 0)).toMatchObject({ messages: [{ role: 'user', content: 'hi there' }], stream: true, stream_options: { include_usage: true } });
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}');
    expect(saved.conversations[0].messages[1].content).toBe('Hello **world**');
  });

  it('counts tokens read and written: estimated while streaming, real once usage arrives, summed per chat', async () => {
    const usage = (p: number, c: number) => `data: ${JSON.stringify({ choices: [], usage: { prompt_tokens: p, completion_tokens: c } })}\n\n`;
    let n = 0;
    mockFetch((_url, init) => {
      n += 1;
      return n === 1 ? heldReply(init) : sseResponse([delta('ok'), FINISH, usage(30, 5), DONE]);
    });
    renderApp();
    typeAndSend('x'.repeat(35));
    await screen.findByText('partial');
    const live = screen.getAllByLabelText('Tokens used by this reply').pop();
    expect(live?.textContent).toBe('Read ≈10 · written ≈1 tokens · ≈ means estimated');
    fireEvent.click(screen.getByRole('button', { name: 'Stop' }));
    await screen.findByText('Stopped.');
    // A stopped reply is still counted, as an estimate.
    const totals = await screen.findByLabelText('Totals for this chat');
    expect(boxValue(totals, 'Read')).toBe('≈10');
    expect(boxValue(totals, 'Written')).toBe('≈1');
    expect(screen.getByLabelText('This chat').textContent).toContain('≈ means estimated');

    typeAndSend('again');
    await waitFor(() => expect(boxValue(screen.getByLabelText('Totals for this chat'), 'Read')).toBe('≈40'));
    expect(boxValue(screen.getByLabelText('Totals for this chat'), 'Written')).toBe('≈6');
    // Real usage replaces the estimate for that reply; the chat total stays marked while one estimate remains.
    const replies = screen.getAllByLabelText('Tokens used by this reply');
    expect(replies[replies.length - 1]?.textContent).toBe('Read 30 · written 5 tokens');
  });

  it('shows a chat total without a mark when every reply has real usage, in Spanish', async () => {
    mockFetch(() => sseResponse([delta('Hola'), FINISH, USAGE, DONE]));
    renderApp(['es-ES']);
    const box = screen.getByLabelText('Mensaje');
    fireEvent.change(box, { target: { value: 'hola' } });
    fireEvent.keyDown(box, { key: 'Enter' });
    const totals = await screen.findByLabelText('Totales de este chat');
    expect(boxValue(totals, 'Leídos')).toBe('12');
    expect(boxValue(totals, 'Escritos')).toBe('3');
    expect(screen.getByLabelText('Este chat').textContent).not.toContain('≈');
    expect(screen.getByLabelText('Tokens usados por esta respuesta').textContent).toBe('Leídos 12 · escritos 3 tokens');
  });

  it('shows the context as one bar of boxes with their numbers', async () => {
    mockFetch(() => sseResponse([delta('Hola'), FINISH, USAGE, DONE]), () => modelsResponse({ context_length: 100 }));
    renderApp();
    await screen.findByText('Context: 100 tokens');
    expect(screen.queryByLabelText('This chat')).toBeNull();
    typeAndSend('hi');
    const bar = await screen.findByRole('img', { name: 'Context: prompt 12 tokens, last reply 3, free 85 of 100 (15% full)' });
    expect(within(bar).getByText('Prompt').nextElementSibling?.textContent).toBe('12');
    expect(within(bar).getByText('Last reply').nextElementSibling?.textContent).toBe('3');
    expect(within(bar).getByText('Free').nextElementSibling?.textContent).toBe('85');
    expect(screen.getByLabelText('This chat').textContent).toContain('15% of 100 tokens');
  });

  it('keeps the chat list usable while a reply is being written', async () => {
    let n = 0;
    mockFetch((_u, init) => (++n === 1 ? sseResponse([delta('first answer'), FINISH, USAGE, DONE]) : heldReply(init)));
    renderApp();
    typeAndSend('one');
    await screen.findByText('first answer');
    fireEvent.click(screen.getByRole('button', { name: 'New chat' }));
    typeAndSend('two');
    await screen.findByText('partial');
    // The reply to "two" is running: starting and switching chats are open, deleting waits.
    expect((screen.getByRole('button', { name: 'New chat' }) as HTMLButtonElement).disabled).toBe(false);
    const first = screen.getByRole('button', { name: /^one/ }) as HTMLButtonElement;
    expect(first.disabled).toBe(false);
    expect((screen.getByRole('button', { name: 'Delete “one”' }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole('button', { name: 'Delete all' }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(first);
    expect(await screen.findByText('first answer')).toBeTruthy();
    expect(screen.getByText('A reply is being written in another chat. You can read this one; sending waits until it ends.')).toBeTruthy();
    expect(screen.queryByText('partial')).toBeNull();
    // The running chat says so in the list, and its reply is still there when you come back.
    const second = screen.getByRole('button', { name: /^two/ });
    expect(second.textContent).toContain('writing…');
    fireEvent.click(second);
    expect(await screen.findByText('partial')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Stop' }));
    await screen.findByText('Stopped.');
  });

  describe('retry', () => {
    it('offers Retry on a failed reply and asks again from the same point once the server is back', async () => {
      let n = 0;
      const fetchMock = mockFetch(() => {
        if (++n === 1) throw new TypeError('Failed to fetch');
        return sseResponse([delta('back again'), FINISH, USAGE, DONE]);
      });
      renderApp();
      typeAndSend('hello');
      expect(await screen.findByText('Could not reach the server.')).toBeTruthy();
      fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
      expect(await screen.findByText('back again')).toBeTruthy();
      expect(screen.queryByText('Could not reach the server.')).toBeNull();
      expect(screen.queryByRole('button', { name: 'Retry' })).toBeNull();
      expect(bodyOf(fetchMock, 1).messages).toEqual(bodyOf(fetchMock, 0).messages);
      // The failed attempt was replaced, not kept next to the answer.
      expect(JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}').conversations[0].messages).toHaveLength(2);
    });

    it('fails again with Retry still there when the server is still down', async () => {
      mockFetch(() => {
        throw new TypeError('Failed to fetch');
      });
      renderApp();
      typeAndSend('hello');
      fireEvent.click(await screen.findByRole('button', { name: 'Retry' }));
      await waitFor(() => expect(screen.getAllByText('Could not reach the server.')).toHaveLength(1));
      expect(await screen.findByRole('button', { name: 'Retry' })).toBeTruthy();
    });

    it('is not offered for a reply the user stopped, nor while a reply is running, nor on an older failure', async () => {
      let n = 0;
      mockFetch((_u, init) => {
        n += 1;
        if (n === 1) throw new TypeError('Failed to fetch');
        return n === 2 ? sseResponse([delta('ok'), FINISH, DONE]) : heldReply(init);
      });
      renderApp();
      typeAndSend('one');
      await screen.findByRole('button', { name: 'Retry' });
      typeAndSend('two');
      await screen.findByText('ok');
      // "one" failed earlier, but only the last message of the chat gets the button.
      expect(screen.queryByRole('button', { name: 'Retry' })).toBeNull();
      typeAndSend('three');
      await screen.findByText('partial');
      expect(screen.queryByRole('button', { name: 'Retry' })).toBeNull();
      fireEvent.click(screen.getByRole('button', { name: 'Stop' }));
      await screen.findByText('Stopped.');
      expect(screen.queryByRole('button', { name: 'Retry' })).toBeNull();
    });
  });

  describe('model picker', () => {
    const twoModels = () =>
      new Response(
        JSON.stringify({
          data: [
            { id: 'qwen3.5-4b', context_length: 8192 },
            { id: 'whiterabbitneo-2.5-qwen2.5-coder-7b', context_length: 32768 },
          ],
        }),
        { status: 200 },
      );

    it('lists every model with its context length and sends the chosen one, per conversation', async () => {
      const fetchMock = mockFetch(() => sseResponse([delta('ok'), FINISH, DONE]), twoModels);
      renderApp();
      const select = (await screen.findByLabelText('Model for this chat')) as HTMLSelectElement;
      expect([...select.options].map((o) => o.textContent)).toEqual([
        'qwen3.5-4b · 8,192 tokens of context',
        'whiterabbitneo-2.5-qwen2.5-coder-7b · 32,768 tokens of context',
      ]);
      expect(select.value).toBe('qwen3.5-4b');
      expect(screen.getByText('Context: 8,192 tokens')).toBeTruthy();

      fireEvent.change(select, { target: { value: 'whiterabbitneo-2.5-qwen2.5-coder-7b' } });
      expect(screen.getByText('Context: 32,768 tokens')).toBeTruthy();
      typeAndSend('one');
      await screen.findByText('ok');
      await waitFor(() => expect(screen.getByRole('button', { name: 'Send' })).toBeTruthy());
      expect(bodyOf(fetchMock, 0).model).toBe('whiterabbitneo-2.5-qwen2.5-coder-7b');
      expect(JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}').conversations[0].model).toBe('whiterabbitneo-2.5-qwen2.5-coder-7b');

      // The choice belongs to this conversation: a new chat starts on the pick for new chats.
      fireEvent.click(screen.getByRole('button', { name: 'New chat' }));
      fireEvent.change(await screen.findByLabelText('Model for this chat'), { target: { value: 'qwen3.5-4b' } });
      typeAndSend('two');
      await waitFor(() => expect(fetchMock.mock.calls.filter(([u]) => String(u).endsWith('/completions'))).toHaveLength(2));
      expect(bodyOf(fetchMock, 1).model).toBe('qwen3.5-4b');
      fireEvent.click(screen.getByRole('button', { name: /^one/ }));
      await waitFor(() => expect((screen.getByLabelText('Model for this chat') as HTMLSelectElement).value).toBe('whiterabbitneo-2.5-qwen2.5-coder-7b'));
    });

    it('shows no picker and sends no model for a server that lists one model', async () => {
      const fetchMock = mockFetch(() => sseResponse([delta('ok'), FINISH, DONE]));
      renderApp();
      await screen.findByText('Model: qwen3.5-4b');
      expect(screen.queryByLabelText('Model for this chat')).toBeNull();
      typeAndSend('hi');
      await screen.findByText('ok');
      expect(bodyOf(fetchMock, 0)).not.toHaveProperty('model');
    });

    it('is in Spanish and is disabled while a reply streams', async () => {
      mockFetch((_url, init) => heldReply(init), twoModels);
      renderApp(['es-ES']);
      const select = (await screen.findByLabelText('Modelo de este chat')) as HTMLSelectElement;
      expect(select.options[1]?.textContent).toBe('whiterabbitneo-2.5-qwen2.5-coder-7b · 32.768 tokens de contexto');
      const box = screen.getByLabelText('Mensaje');
      fireEvent.change(box, { target: { value: 'hola' } });
      fireEvent.keyDown(box, { key: 'Enter' });
      await screen.findByText('partial');
      expect(select.disabled).toBe(true);
    });
  });

  describe('compacting', () => {
    const SUMMARY_TEXT = 'The user asked about apples and pears.';
    const PAD = ' lorem ipsum dolor sit amet';
    /** A saved chat of `count` turns; the second reply was stopped part way. */
    function seed(count: number, model: string | null = null) {
      const messages = Array.from({ length: count }, (_, i) => [
        { id: `u${i}`, role: 'user', content: `question ${i}${PAD}`, typed: `question ${i}${PAD}`, attachmentNames: [], createdAt: 1 },
        { id: `a${i}`, role: 'assistant', content: i === 1 ? `partial answ${PAD}` : `answer ${i}${PAD}`, reasoning: '', status: i === 1 ? 'stopped' : 'done', error: null, stats: null, createdAt: 1 },
      ]).flat();
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({ version: 1, activeId: 'c', settings: {}, conversations: [{ id: 'c', title: 'Fruit', createdAt: 1, updatedAt: 1, model, messages }] }),
      );
    }
    const isSummary = (init?: RequestInit) => String(JSON.parse(String(init?.body)).messages[0].content).startsWith('You compress');
    const reply = (init: RequestInit | undefined, text = 'ok') =>
      isSummary(init) ? sseResponse([delta(SUMMARY_TEXT), FINISH, DONE]) : sseResponse([delta(text), FINISH, DONE]);

    it('the Compact button replaces older turns with a summary, keeps the last two turns and the old messages on screen', async () => {
      seed(4);
      const fetchMock = mockFetch((_u, init) => reply(init));
      renderApp();
      fireEvent.click(await screen.findByRole('button', { name: 'Compact chat' }));
      expect(await screen.findByText('Compacted: 4 earlier messages replaced by a summary')).toBeTruthy();
      const summaryBody = bodyOf(fetchMock, 0);
      expect(summaryBody.messages[1].content).toContain('question 0');
      // The stopped partial reply is in what was summarised, marked as stopped, never dropped silently.
      expect(summaryBody.messages[1].content).toContain('partial answ lorem');
      expect(summaryBody.messages[1].content).toContain('[this reply was stopped by the user before it ended]');
      // Nothing is removed from the saved chat.
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}').conversations[0].messages;
      expect(saved).toHaveLength(9);
      expect(saved[4]).toMatchObject({ role: 'summary', content: SUMMARY_TEXT, covered: 4, auto: false });
      expect(screen.getByText(`question 0${PAD}`)).toBeTruthy();

      typeAndSend('next');
      await waitFor(() => expect(fetchMock.mock.calls.filter(([u]) => String(u).endsWith('/completions'))).toHaveLength(2));
      const body = bodyOf(fetchMock, 1);
      expect(body.messages[0]).toMatchObject({ role: 'system' });
      expect(body.messages[0].content).toContain(SUMMARY_TEXT);
      expect(body.messages.slice(1).map((m: { content: string }) => m.content)).toEqual([`question 2${PAD}`, `answer 2${PAD}`, `question 3${PAD}`, `answer 3${PAD}`, 'next']);
    });

    it('is off when there is little to compact or a reply is streaming', async () => {
      seed(2);
      mockFetch((_u, init) => heldReply(init));
      renderApp();
      expect((await screen.findByRole('button', { name: 'Compact chat' })).hasAttribute('disabled')).toBe(true);
    });

    it('compacts by itself before sending when the next request would fill the context', async () => {
      seed(4);
      const fetchMock = mockFetch(
        (_u, init) => reply(init, 'sure'),
        () => modelsResponse({ context_length: 100 }),
      );
      renderApp();
      await screen.findByText('Context: 100 tokens');
      typeAndSend('x');
      expect(await screen.findByText('sure')).toBeTruthy();
      const chats = fetchMock.mock.calls.filter(([u]) => String(u).endsWith('/completions'));
      expect(chats).toHaveLength(2);
      expect(isSummary(chats[0]?.[1])).toBe(true);
      expect(bodyOf(fetchMock, 1).messages[0].content).toContain(SUMMARY_TEXT);
      expect(screen.getByText('Compacted: 4 earlier messages replaced by a summary')).toBeTruthy();
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}').conversations[0].messages;
      expect(saved.find((m: { role: string }) => m.role === 'summary')).toMatchObject({ auto: true });
    });

    it('does not compact by itself while the chat fits, or when the server gives no context length', async () => {
      seed(4);
      const fetchMock = mockFetch((_u, init) => reply(init, 'fine'));
      renderApp();
      await screen.findByText('Model: qwen3.5-4b');
      typeAndSend('x');
      await screen.findByText('fine');
      expect(fetchMock.mock.calls.filter(([u]) => String(u).endsWith('/completions'))).toHaveLength(1);
    });

    it('keeps every message and says why when the summary fails, then still sends', async () => {
      seed(4);
      const fetchMock = mockFetch(
        (_u, init) => (isSummary(init) ? sseResponse([FINISH, DONE]) : sseResponse([delta('sure'), FINISH, DONE])),
        () => modelsResponse({ context_length: 100 }),
      );
      renderApp();
      await screen.findByText('Context: 100 tokens');
      typeAndSend('x');
      expect(await screen.findByText('sure')).toBeTruthy();
      expect(screen.getByText('The chat was not compacted: The model wrote no summary.')).toBeTruthy();
      // A failed compaction can be retried from its message.
      expect(within(screen.getByText(/The chat was not compacted/)).getByRole('button', { name: 'Retry' })).toBeTruthy();
      expect(screen.queryByText(/Compacted:/)).toBeNull();
      expect(bodyOf(fetchMock, 1).messages).toHaveLength(9);
    });

    it('keeps the failed-compaction banner and its Retry with the chat that failed', async () => {
      seed(4);
      mockFetch(
        (_u, init) => (isSummary(init) ? sseResponse([FINISH, DONE]) : sseResponse([delta('sure'), FINISH, DONE])),
        () => modelsResponse({ context_length: 100 }),
      );
      renderApp();
      await screen.findByText('Context: 100 tokens');
      typeAndSend('x');
      await screen.findByText('The chat was not compacted: The model wrote no summary.');
      await screen.findByText('sure');
      fireEvent.click(screen.getByRole('button', { name: 'New chat' }));
      expect(screen.queryByText(/The chat was not compacted/)).toBeNull();
      fireEvent.click(screen.getByRole('button', { name: /^Fruit/ }));
      expect(screen.getByText(/The chat was not compacted/)).toBeTruthy();
    });

    it('does not compact by itself when Auto compact is off, but the button still does, and the choice is kept', async () => {
      seed(4);
      const fetchMock = mockFetch(
        (_u, init) => reply(init, 'sure'),
        () => modelsResponse({ context_length: 100 }),
      );
      renderApp();
      await screen.findByText('Context: 100 tokens');
      const auto = screen.getByRole('checkbox', { name: 'Auto compact' }) as HTMLInputElement;
      expect(auto.checked).toBe(true);
      expect(document.getElementById('compact-hint')?.textContent).toContain('reaches 80% of the context');
      fireEvent.click(auto);
      expect(JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}').settings.autoCompact).toBe(false);
      expect(document.getElementById('compact-hint')?.textContent).toMatch(/^Manual: nothing is summarised/);
      // Past 80% of the window the automatic mode would summarise first; here the chat is sent whole.
      typeAndSend('x');
      expect(await screen.findByText('sure')).toBeTruthy();
      const chats = fetchMock.mock.calls.filter(([u]) => String(u).endsWith('/completions'));
      expect(chats).toHaveLength(1);
      expect(isSummary(chats[0]?.[1])).toBe(false);
      expect(bodyOf(fetchMock, 0).messages).toHaveLength(9);
      expect(screen.queryByText(/Compacted:/)).toBeNull();
      fireEvent.click(screen.getByRole('button', { name: 'Compact chat' }));
      expect(await screen.findByText('Compacted: 6 earlier messages replaced by a summary')).toBeTruthy();
      expect(screen.getByText('Compacted: 6 earlier messages replaced by a summary').closest('details')?.textContent).toContain('Made with the Compact button.');
    });

    it('stopping while it summarises leaves the chat as it was', async () => {
      seed(4);
      mockFetch((_u, init) => heldReply(init));
      renderApp();
      fireEvent.click(await screen.findByRole('button', { name: 'Compact chat' }));
      fireEvent.click(await screen.findByRole('button', { name: 'Stop' }));
      await waitFor(() => expect((screen.getByRole('button', { name: 'Compact chat' }) as HTMLButtonElement).disabled).toBe(false));
      expect(screen.queryByText(/Compacted:/)).toBeNull();
      expect(screen.queryByRole('alert')).toBeNull();
      expect(JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}').conversations[0].messages).toHaveLength(8);
    });

    it('is in Spanish', async () => {
      seed(4);
      mockFetch((_u, init) => reply(init));
      renderApp(['es-ES']);
      fireEvent.click(await screen.findByRole('button', { name: 'Compactar chat' }));
      expect(await screen.findByText('Compactado: 4 mensajes anteriores sustituidos por un resumen')).toBeTruthy();
    });
  });

  describe('tools', () => {
    function enableTools(maxSteps = 8) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ version: 1, activeId: null, conversations: [], settings: { tools: true, maxSteps } }));
    }
    const asks = (id = 'c1', expr = '6*7') => sseResponse([toolCallDelta(id, 'calculator', JSON.stringify({ expression: expr })), FINISH_TOOLS, DONE]);

    it('runs the tool the model asks for, shows a collapsible trace and sends the result back', async () => {
      enableTools();
      let n = 0;
      const fetchMock = mockFetch(() => (++n === 1 ? asks() : sseResponse([delta('It is 42.'), FINISH, DONE])));
      renderApp();
      typeAndSend('what is 6*7?');
      expect(await screen.findByText('It is 42.')).toBeTruthy();
      const summary = screen.getByText('Tool calls: calculator');
      const trace = summary.closest('details') as HTMLDetailsElement;
      expect(trace.open).toBe(false);
      expect(within(trace).getByText('{"expression":"6*7"}')).toBeTruthy();
      expect(within(trace).getByText('42')).toBeTruthy();
      expect(bodyOf(fetchMock, 0)).toMatchObject({ tool_choice: 'auto', tools: [{ function: { name: 'calculator' } }, { function: { name: 'current_time' } }] });
      expect(bodyOf(fetchMock, 1).messages.slice(-2)).toMatchObject([{ role: 'assistant', content: null, tool_calls: [{ id: 'c1' }] }, { role: 'tool', tool_call_id: 'c1', content: '42' }]);
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}').conversations[0].messages;
      expect(saved.map((m: { role: string }) => m.role)).toEqual(['user', 'assistant', 'tool', 'assistant']);
      // The next question re-sends the whole exchange, calls and results included.
      await waitFor(() => expect(screen.getByRole('button', { name: 'Send' })).toBeTruthy());
      typeAndSend('thanks');
      await waitFor(() => expect(fetchMock.mock.calls.filter(([u]) => String(u).endsWith('/completions'))).toHaveLength(3));
      expect(bodyOf(fetchMock, 2).messages.map((m: { role: string }) => m.role)).toEqual(['user', 'assistant', 'tool', 'assistant', 'user']);
    });

    it('compacts between steps when the server’s own count says the context is nearly full', async () => {
      const turns = Array.from({ length: 4 }, (_, i) => [
        { id: `u${i}`, role: 'user', content: `q${i}`, typed: `q${i}`, attachmentNames: [], createdAt: 1 },
        { id: `a${i}`, role: 'assistant', content: `a${i}`, reasoning: '', status: 'done', error: null, stats: null, createdAt: 1 },
      ]).flat();
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({ version: 1, activeId: 'c', settings: { tools: true }, conversations: [{ id: 'c', title: 't', createdAt: 1, updatedAt: 1, model: null, messages: turns }] }),
      );
      const usage = `data: ${JSON.stringify({ choices: [], usage: { prompt_tokens: 90, completion_tokens: 5 } })}\n\n`;
      let n = 0;
      const fetchMock = mockFetch(
        (_u, init) => {
          const system = String(JSON.parse(String(init?.body)).messages[0].content);
          if (system.startsWith('You compress')) return sseResponse([delta('SUM'), FINISH, DONE]);
          return ++n === 1
            ? sseResponse([toolCallDelta('c1', 'calculator', '{"expression":"6*7"}'), FINISH_TOOLS, usage, DONE])
            : sseResponse([delta('It is 42.'), FINISH, DONE]);
        },
        () => modelsResponse({ context_length: 100 }),
      );
      renderApp();
      await screen.findByText('Context: 100 tokens');
      typeAndSend('6*7?');
      expect(await screen.findByText('It is 42.')).toBeTruthy();
      expect(screen.getByText('Compacted: 6 earlier messages replaced by a summary')).toBeTruthy();
      const bodies = fetchMock.mock.calls.filter(([u]) => String(u).endsWith('/completions')).map((c) => JSON.parse(String(c[1]?.body)));
      expect(bodies).toHaveLength(3);
      expect(String(bodies[1].messages[0].content)).toContain('You compress');
      expect(String(bodies[2].messages[0].content)).toContain('SUM');
      expect(bodies[2].messages.slice(-2).map((m: { role: string }) => m.role)).toEqual(['assistant', 'tool']);
      expect(JSON.stringify(bodies[2].messages)).not.toContain('q0');
    });

    it('stops at the step cap with a note, and the last request forbids tool calls', async () => {
      enableTools(2);
      const fetchMock = mockFetch(() => asks());
      renderApp();
      typeAndSend('loop');
      expect(await screen.findByText('Stopped at the limit of 2 steps; the tool calls above were not run.')).toBeTruthy();
      const chats = fetchMock.mock.calls.filter(([u]) => String(u).endsWith('/completions'));
      expect(chats).toHaveLength(2);
      expect(bodyOf(fetchMock, 1).tool_choice).toBe('none');
      expect(screen.getByText('Not run')).toBeTruthy();
      expect(screen.getByRole('button', { name: 'Send' })).toBeTruthy();
    });

    it('can be stopped with the Stop button between the steps of a long run', async () => {
      enableTools();
      let n = 0;
      mockFetch((_u, init) => (++n === 1 ? asks() : heldReply(init)));
      renderApp();
      typeAndSend('go');
      await screen.findByText('partial');
      fireEvent.click(screen.getByRole('button', { name: 'Stop' }));
      expect(await screen.findByText('Stopped.')).toBeTruthy();
      expect(screen.getByText('Tool calls: calculator')).toBeTruthy();
      expect(n).toBe(2);
    });

    it('does not offer tools unless they are turned on in Settings, and the cap is set there', async () => {
      const fetchMock = mockFetch(() => sseResponse([delta('hi'), FINISH, DONE]));
      renderApp();
      typeAndSend('x');
      await screen.findByText('hi');
      expect(bodyOf(fetchMock, 0)).not.toHaveProperty('tools');
      fireEvent.click(screen.getByRole('button', { name: 'Settings' }));
      const box = screen.getByLabelText('Let the model use tools') as HTMLInputElement;
      const steps = screen.getByLabelText('Most steps per message') as HTMLInputElement;
      expect(box.checked).toBe(false);
      expect(steps.disabled).toBe(true);
      fireEvent.click(box);
      expect(steps.disabled).toBe(false);
      expect(steps.value).toBe('8');
      fireEvent.change(steps, { target: { value: '3' } });
      fireEvent.change(steps, { target: { value: '99' } });
      expect(steps.value).toBe('3');
      await waitFor(() => expect(JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}').settings).toMatchObject({ tools: true, maxSteps: 3 }));
    });

    it('shows a rejected tool request in the user’s language', async () => {
      enableTools();
      mockFetch(() => new Response(JSON.stringify({ error: { message: 'tools must be an array', type: 'invalid_request_error', code: 'invalid_tools' } }), { status: 400 }));
      renderApp(['es-ES']);
      const box = screen.getByLabelText('Mensaje');
      fireEvent.change(box, { target: { value: 'hola' } });
      fireEvent.keyDown(box, { key: 'Enter' });
      expect((await screen.findByRole('alert')).textContent).toBe('El servidor rechazó la petición de herramientas: tools must be an array');
    });

    it('labels the trace in Spanish and has no accessibility violations', async () => {
      enableTools();
      let n = 0;
      mockFetch(() => (++n === 1 ? asks() : sseResponse([delta('Son 42.'), FINISH, DONE])));
      const { container } = renderApp(['es-ES']);
      const box = screen.getByLabelText('Mensaje');
      fireEvent.change(box, { target: { value: 'cuánto es' } });
      fireEvent.keyDown(box, { key: 'Enter' });
      await screen.findByText('Son 42.');
      expect(screen.getByText('Llamadas a herramientas: calculator')).toBeTruthy();
      expect(await axeViolations(container)).toEqual([]);
    });
  });

  it('re-sends the previous answer exactly as received', async () => {
    const fetchMock = mockFetch(() => sseResponse([delta('A *b*\n\n'), delta('`c`'), FINISH, DONE]));
    renderApp();
    typeAndSend('one');
    await screen.findByText('c');
    await waitFor(() => expect(screen.getByRole('button', { name: 'Send' })).toBeTruthy());
    typeAndSend('two');
    await waitFor(() => expect(fetchMock.mock.calls.filter(([u]) => String(u).endsWith('/completions'))).toHaveLength(2));
    expect(bodyOf(fetchMock, 1).messages).toEqual([
      { role: 'user', content: 'one' },
      { role: 'assistant', content: 'A *b*\n\n`c`' },
      { role: 'user', content: 'two' },
    ]);
  });

  it('stops a reply with the Stop button and with Esc', async () => {
    mockFetch((_url, init) => heldReply(init));
    renderApp();
    typeAndSend('long one');
    await screen.findByText('partial');
    fireEvent.click(screen.getByRole('button', { name: 'Stop' }));
    expect(await screen.findByText('Stopped.')).toBeTruthy();

    typeAndSend('again');
    await waitFor(() => expect(screen.getAllByText('partial')).toHaveLength(2));
    fireEvent.keyDown(window, { key: 'Escape' });
    await waitFor(() => expect(screen.getAllByText('Stopped.')).toHaveLength(2));
  });

  it('shows a server error localized by its code', async () => {
    const body = { error: { message: 'image too large', type: 'invalid_request_error', code: 'image_too_large' } };
    mockFetch(() => new Response(JSON.stringify(body), { status: 400 }));
    renderApp(['es-ES']);
    const box = screen.getByLabelText('Mensaje');
    fireEvent.change(box, { target: { value: 'hola' } });
    fireEvent.keyDown(box, { key: 'Enter' });
    expect((await screen.findByRole('alert')).textContent).toBe('Una imagen es demasiado grande para el servidor.');
  });

  it('uses the catalog for a context overflow instead of the server sentence', async () => {
    const body = {
      error: {
        message: 'the prompt is 80 tokens and does not fit the 64-token context',
        type: 'invalid_request_error',
        code: 'context_length_exceeded',
      },
    };
    mockFetch(() => new Response(JSON.stringify(body), { status: 400 }));
    renderApp(['es-ES']);
    const box = screen.getByLabelText('Mensaje');
    fireEvent.change(box, { target: { value: 'hola' } });
    fireEvent.keyDown(box, { key: 'Enter' });
    const alert = await screen.findByRole('alert');
    expect(alert.textContent).toBe('La conversación ya no cabe en el contexto del modelo. Empieza un chat nuevo o envía menos.');
    expect(alert.textContent).not.toContain('does not fit');
  });

  it('uses the English catalog for the same context overflow', async () => {
    const body = {
      error: {
        message: 'the prompt is 80 tokens and does not fit the 64-token context',
        type: 'invalid_request_error',
        code: 'context_length_exceeded',
      },
    };
    mockFetch(() => new Response(JSON.stringify(body), { status: 400 }));
    renderApp();
    typeAndSend('hi');
    const alert = await screen.findByRole('alert');
    expect(alert.textContent).toBe('The conversation no longer fits the model’s context. Start a new chat or send less.');
    expect(alert.textContent).not.toContain('does not fit');
  });

  it('localizes a failed model start with the catalog', async () => {
    mockFetch((url) => {
      if (String(url).endsWith('/control/serve')) {
        return new Response(JSON.stringify({ ok: false, code: 'start_failed', error: 'lm-serve did not become healthy in time' }), { status: 502 });
      }
      return new Response('no', { status: 500 });
    });
    renderApp(['es-ES']);
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    fireEvent.click(await screen.findByRole('button', { name: 'Ajustes' }));
    fireEvent.click(screen.getByRole('button', { name: 'Arrancar Qwen3.5-4B' }));
    const alert = await screen.findByRole('alert');
    expect(alert.textContent).toBe('No se pudo arrancar el modelo en el servidor.');
    expect(alert.textContent).not.toContain('healthy');
  });

  it('shows the server message for an unknown error code', async () => {
    mockFetch(() => new Response(JSON.stringify({ error: { message: 'tools are not supported', type: 'invalid_request_error' } }), { status: 400 }));
    renderApp();
    typeAndSend('x');
    expect((await screen.findByRole('alert')).textContent).toBe('The server answered 400: tools are not supported');
  });

  it('saves the sent message before the reply finishes', async () => {
    mockFetch((_url, init) => heldReply(init));
    renderApp();
    typeAndSend('keep me');
    await screen.findByText('partial');
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}');
    expect(saved.conversations[0].messages.map((m: { typed?: string }) => m.typed)).toEqual(['keep me']);
    fireEvent.click(screen.getByRole('button', { name: 'Stop' }));
    await screen.findByText('Stopped.');
  });

  it('localizes proxy errors and errors sent inside a stream', async () => {
    let call = 0;
    mockFetch(() => {
      call += 1;
      return call === 1
        ? new Response(JSON.stringify({ error: { message: 'lm-serve is not reachable', type: 'proxy_error', code: 'upstream_unreachable' } }), { status: 502 })
        : sseResponse([delta('a'), 'data: {"error":{"message":"generation failed","type":"server_error","code":null}}\n\n']);
    });
    renderApp(['es']);
    const box = screen.getByLabelText('Mensaje');
    fireEvent.change(box, { target: { value: 'uno' } });
    fireEvent.keyDown(box, { key: 'Enter' });
    expect((await screen.findByRole('alert')).textContent).toBe('El proxy no puede conectar con lm-serve. ¿Está en marcha?');
    await screen.findByRole('button', { name: 'Enviar' });
    fireEvent.change(box, { target: { value: 'dos' } });
    fireEvent.keyDown(box, { key: 'Enter' });
    await waitFor(() => expect(screen.getAllByRole('alert')).toHaveLength(2));
    expect(screen.getAllByRole('alert')[1]?.textContent).toBe('El servidor detuvo la respuesta con un error: generation failed');
  });

  it('attaches two files that share a name', async () => {
    mockFetch(() => sseResponse([FINISH]));
    renderApp();
    const files = [new File(['one'], 'notes.txt'), new File(['two'], 'notes.txt')];
    fireEvent.change(screen.getByTestId('file-input'), { target: { files } });
    const list = await screen.findByRole('list', { name: 'Attachments' });
    await waitFor(() => expect(within(list).getAllByText('notes.txt')).toHaveLength(2));
    expect((screen.getByRole('button', { name: 'Send' }) as HTMLButtonElement).disabled).toBe(false);
  });

  it('renders raw HTML in a reply as text', async () => {
    mockFetch(() => sseResponse([delta('<script>alert(1)</script> and <b>bold</b>'), FINISH, DONE]));
    const { container } = renderApp();
    typeAndSend('xss');
    expect(await screen.findByText(/<script>alert\(1\)<\/script>/)).toBeTruthy();
    expect(container.querySelector('.md script, .md b')).toBeNull();
  });

  it('refuses a file it cannot read', async () => {
    mockFetch(() => sseResponse([FINISH]));
    renderApp();
    const input = screen.getByTestId('file-input');
    fireEvent.change(input, { target: { files: [new File(['x'], 'song.mp3', { type: 'audio/mpeg' })] } });
    expect((await screen.findByRole('alert')).textContent).toContain('song.mp3: this kind of file is not supported.');
    fireEvent.change(input, { target: { files: [new File(['a\u0000b'], 'data.txt', { type: 'text/plain' })] } });
    expect((await screen.findByRole('alert')).textContent).toContain('data.txt: not a text file.');
  });

  it('attaches a text file as a fenced block with an estimate', async () => {
    const fetchMock = mockFetch(() => sseResponse([delta('ok'), FINISH, DONE]));
    renderApp();
    fireEvent.change(screen.getByTestId('file-input'), { target: { files: [new File(['print(1)\n'], 'a.py')] } });
    const list = await screen.findByRole('list', { name: 'Attachments' });
    expect(within(list).getByText('a.py')).toBeTruthy();
    expect(within(list).getByText(/tokens \(estimate\)/)).toBeTruthy();
    typeAndSend('explain');
    await screen.findByText('ok');
    expect(bodyOf(fetchMock, 0).messages[0].content).toBe('a.py\n```python\nprint(1)\n```\n\nexplain');
    expect(screen.getByText('Attached: a.py')).toBeTruthy();
  });

  it('disables image upload and says why when the server does not report vision', async () => {
    mockFetch(() => sseResponse([FINISH]));
    renderApp();
    await screen.findByText('Model: qwen3.5-4b');
    const button = screen.getByRole('button', { name: 'Attach image' }) as HTMLButtonElement;
    expect(button.disabled).toBe(true);
    expect(button.getAttribute('aria-describedby')).toBe('image-reason');
    expect(document.getElementById('image-reason')?.textContent).toBe('Images: off, because this server does not say whether it can read them.');
  });

  it('disables image upload when vision is unavailable', async () => {
    mockFetch(() => sseResponse([FINISH]), () => modelsResponse({ capabilities: { vision: { status: 'unavailable' } } }));
    renderApp();
    await screen.findByText('Model: qwen3.5-4b');
    const button = screen.getByRole('button', { name: 'Attach image' }) as HTMLButtonElement;
    expect(button.disabled).toBe(true);
    expect(document.getElementById('image-reason')?.textContent).toBe('Images: this model cannot read images.');
  });

  it('enables image upload when the server reports vision ok', async () => {
    mockFetch(() => sseResponse([FINISH]), () => modelsResponse({ context_length: 8192, capabilities: { vision: { status: 'ok', max_pixels: 1048576, pixels_per_token: 1024 } } }));
    renderApp();
    await screen.findByText(/Context: 8,192 tokens/);
    await waitFor(() => expect((screen.getByRole('button', { name: 'Attach image' }) as HTMLButtonElement).disabled).toBe(false));
  });

  it('refuses to send what would overflow the context', async () => {
    mockFetch(() => sseResponse([FINISH]), () => modelsResponse({ context_length: 10 }));
    renderApp();
    await screen.findByText(/Context: 10 tokens/);
    fireEvent.change(screen.getByLabelText('Message'), { target: { value: 'x'.repeat(100) } });
    expect((await screen.findByRole('alert')).textContent).toContain('the context holds 10');
    expect((screen.getByRole('button', { name: 'Send' }) as HTMLButtonElement).disabled).toBe(true);
  });

  it('shows when the server is unreachable and retries', async () => {
    let up = false;
    mockFetch(() => sseResponse([FINISH]), () => (up ? modelsResponse() : Promise.reject(new TypeError('down')) as unknown as Response));
    renderApp();
    expect(await screen.findByText('Could not reach the server.')).toBeTruthy();
    up = true;
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
    expect(await screen.findByText('Model: qwen3.5-4b')).toBeTruthy();
  });

  it('renames, deletes and deletes all conversations', async () => {
    const fetchMock = mockFetch(() => sseResponse([delta('ok'), FINISH, DONE]));
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    renderApp();
    typeAndSend('first chat');
    await screen.findByText('ok');
    const nav = screen.getByRole('navigation', { name: 'Conversations' });
    fireEvent.click(within(nav).getByRole('button', { name: 'Rename “first chat”' }));
    fireEvent.change(within(nav).getByLabelText('Name of “first chat”'), { target: { value: 'Renamed' } });
    fireEvent.click(within(nav).getByRole('button', { name: 'Save' }));
    expect(within(nav).getByText('Renamed')).toBeTruthy();

    fireEvent.click(within(nav).getByRole('button', { name: 'New chat' }));
    typeAndSend('second chat');
    await waitFor(() => expect(within(nav).getAllByRole('listitem')).toHaveLength(2));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Send' })).toBeTruthy());
    fireEvent.click(within(nav).getByRole('button', { name: 'Delete “Renamed”' }));
    expect(within(nav).getAllByRole('listitem')).toHaveLength(1);
    expect(fetchMock.mock.calls.some(([url, init]) => String(url).endsWith('/v1/chat/reset') && (init as RequestInit | undefined)?.method === 'POST')).toBe(true);
    fireEvent.click(within(nav).getByRole('button', { name: 'Delete all' }));
    expect(within(nav).getByText('No conversations yet.')).toBeTruthy();
    expect(JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}').conversations).toEqual([]);
  });

  it('deletes an unreadable copy from the banner', () => {
    localStorage.setItem(STORAGE_KEY, '{"version":99,"secret":"keep-me"}');
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    mockFetch(() => sseResponse([FINISH]));
    renderApp();
    const banner = screen.getByRole('alert');
    expect(banner.textContent).toContain('set aside');
    fireEvent.click(within(banner).getByRole('button', { name: 'Delete all' }));
    expect(Object.keys(localStorage).filter((k) => k.startsWith(BACKUP_KEY))).toEqual([]);
    expect(localStorage.getItem(STORAGE_KEY)).not.toContain('keep-me');
  });

  it('discards a locked save so later edits can be stored', () => {
    const data = new Map<string, string>([[STORAGE_KEY, '{"version":99}']]);
    const storage = {
      get length() { return data.size; },
      key: (i: number) => [...data.keys()][i] ?? null,
      getItem: (k: string) => data.get(k) ?? null,
      setItem: (k: string, v: string) => {
        if (k.startsWith(BACKUP_KEY)) throw new DOMException('full', 'QuotaExceededError');
        data.set(k, v);
      },
      removeItem: (k: string) => { data.delete(k); },
      clear: () => { data.clear(); },
    } as Storage;
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    mockFetch(() => sseResponse([FINISH]));
    render(<App storage={storage} />);
    fireEvent.click(within(screen.getByRole('alert')).getByRole('button', { name: 'Delete all' }));
    const saved = JSON.parse(storage.getItem(STORAGE_KEY) ?? '{}') as { version?: number };
    expect(saved.version).toBe(1);
  });

  it('switches language from settings', async () => {
    mockFetch(() => sseResponse([FINISH]));
    renderApp();
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }));
    fireEvent.change(screen.getByLabelText('Language'), { target: { value: 'es' } });
    expect(await screen.findByRole('button', { name: 'Enviar' })).toBeTruthy();
    expect(document.documentElement.lang).toBe('es');
  });

  it('names the conversation toggle in full and shows a short label', () => {
    mockFetch(() => sseResponse([FINISH]));
    renderApp();
    const toggle = screen.getByRole('button', { name: 'Show conversations' });
    expect(toggle.textContent).toBe('Show');
  });

  it('inserts a line from a phone keyboard instead of sending', async () => {
    const fetchMock = mockFetch(() => sseResponse([delta('nope'), FINISH, DONE]));
    vi.stubGlobal('matchMedia', (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener() {},
      removeListener() {},
      addEventListener() {},
      removeEventListener() {},
      dispatchEvent() {
        return false;
      },
    }));
    try {
      renderApp();
      const box = screen.getByLabelText('Message') as HTMLTextAreaElement;
      expect(document.activeElement).not.toBe(box);
      expect(box.placeholder).toBe('Message');
      expect(box.getAttribute('enterkeyhint')).toBe('enter');
      fireEvent.change(box, { target: { value: 'line' } });
      fireEvent.keyDown(box, { key: 'Enter' });
      await Promise.resolve();
      expect(fetchMock.mock.calls.filter(([url]) => String(url).endsWith('/completions'))).toHaveLength(0);
      expect(box.value).toBe('line');
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it('starts gemma with ctx from the header picker after confirm', async () => {
    const fetchMock = mockFetch(
      () => new Response(JSON.stringify({ ok: true, model: 'gemma', health: 'ok' }), { status: 200, headers: { 'Content-Type': 'application/json' } }),
    );
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(true);
    renderApp();
    await screen.findByText('Model: qwen3.5-4b');
    fireEvent.click(screen.getByRole('button', { name: /Choose model and context window/ }));
    fireEvent.click(screen.getByRole('button', { name: 'Gemma 4 E4B' }));
    fireEvent.click(screen.getByRole('radio', { name: '8k' }));
    fireEvent.click(screen.getByRole('button', { name: 'Start' }));
    expect(confirm).toHaveBeenCalledWith('Stop the current model and start Gemma 4 E4B?');
    await waitFor(() => expect(fetchMock.mock.calls.some(([url]) => String(url).endsWith('/control/serve'))).toBe(true));
    const call = fetchMock.mock.calls.find(([url]) => String(url).endsWith('/control/serve'));
    expect(JSON.parse(String(call?.[1]?.body))).toMatchObject({ model: 'gemma', ctx: 8192 });
  });

  it('asks before replacing the running model', async () => {
    const fetchMock = mockFetch(
      () => new Response(JSON.stringify({ ok: true, model: 'gemma', health: 'ok' }), { status: 200, headers: { 'Content-Type': 'application/json' } }),
    );
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(false);
    renderApp();
    await screen.findByText('Model: qwen3.5-4b');
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }));
    const qwen = screen.getByRole('button', { name: 'Start Qwen3.5-4B' });
    expect(qwen.getAttribute('aria-pressed')).toBe('true');
    expect(screen.getByRole('button', { name: 'Start Gemma 4 E4B' }).getAttribute('aria-pressed')).toBe('false');
    fireEvent.click(qwen);
    expect(confirm).toHaveBeenCalledWith('Stop the current model and start Qwen3.5-4B?');
    expect(fetchMock.mock.calls.some(([url]) => String(url).endsWith('/control/serve'))).toBe(false);
    confirm.mockReturnValue(true);
    fireEvent.click(screen.getByRole('button', { name: 'Start Gemma 4 E4B' }));
    await waitFor(() => expect(fetchMock.mock.calls.some(([url]) => String(url).endsWith('/control/serve'))).toBe(true));
    const call = fetchMock.mock.calls.find(([url]) => String(url).endsWith('/control/serve'));
    expect(JSON.parse(String(call?.[1]?.body)).model).toBe('gemma');
  });

  it('shows the live prompt-reading timer before the first token', async () => {
    mockFetch((_url, init) => {
      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          init?.signal?.addEventListener('abort', () => controller.error(new DOMException('aborted', 'AbortError')));
        },
      });
      return new Response(stream);
    });
    renderApp();
    typeAndSend('slow');
    expect(await screen.findByText(/Reading the prompt… \d+ s/)).toBeTruthy();
    expect(within(screen.getByRole('log')).queryByText(/Reading the prompt/)).toBeNull();
    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Stop' })));
  });

  it('keeps focus where it is when a reply ends', async () => {
    mockFetch(() => sseResponse([delta('ok'), FINISH, DONE]));
    renderApp();
    const settings = screen.getByRole('button', { name: 'Settings' });
    settings.focus();
    typeAndSend('hi');
    await screen.findByText('ok');
    expect(document.activeElement).toBe(settings);
  });

  it('returns focus to Rename after cancelling', async () => {
    mockFetch(() => sseResponse([delta('ok'), FINISH, DONE]));
    renderApp();
    typeAndSend('first chat');
    await screen.findByText('ok');
    const nav = screen.getByRole('navigation', { name: 'Conversations' });
    const rename = within(nav).getByRole('button', { name: 'Rename “first chat”' });
    fireEvent.click(rename);
    fireEvent.click(within(nav).getByRole('button', { name: 'Cancel' }));
    await waitFor(() => expect(document.activeElement).toBe(within(nav).getByRole('button', { name: 'Rename “first chat”' })));
  });
});

describe('live scroll stickiness', () => {
  it('stops following while scrolled up, then follows again after Send', async () => {
    type Ctrl = { push: (chunk: string) => void; close: () => void };
    const ctrls: Ctrl[] = [];
    mockFetch((_url, init) => {
      let controller!: ReadableStreamDefaultController<Uint8Array>;
      const enc = new TextEncoder();
      const stream = new ReadableStream<Uint8Array>({
        start(c) {
          controller = c;
          controller.enqueue(enc.encode(delta('start')));
          init?.signal?.addEventListener('abort', () => controller.error(new DOMException('aborted', 'AbortError')));
        },
      });
      ctrls.push({
        push: (chunk) => controller.enqueue(enc.encode(chunk)),
        close: () => controller.close(),
      });
      return new Response(stream, { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
    });
    renderApp();
    typeAndSend('one');
    await screen.findByText('start');

    const log = screen.getByRole('region', { name: 'Conversation' });
    expect(log.tabIndex).toBe(0);
    let scrollTop = 0;
    Object.defineProperty(log, 'clientHeight', { configurable: true, get: () => 400 });
    Object.defineProperty(log, 'scrollHeight', { configurable: true, get: () => 2000 });
    Object.defineProperty(log, 'scrollTop', {
      configurable: true,
      get: () => scrollTop,
      set: (v: number) => {
        scrollTop = v;
      },
    });
    scrollTop = 0;
    fireEvent.scroll(log);

    const first = ctrls[0];
    expect(first).toBeDefined();
    await act(async () => {
      first.push(delta(' more'));
    });
    await screen.findByText(/more/);
    expect(scrollTop).toBe(0);

    await act(async () => {
      first.push(FINISH);
      first.push(DONE);
      first.close();
    });
    await screen.findByRole('button', { name: 'Send' });

    scrollTop = 50;
    typeAndSend('two');
    await screen.findByText('start');
    expect(ctrls).toHaveLength(2);
    // Send re-arms stickiness; the next live token may follow to the bottom.
    expect(scrollTop).toBe(2000);

    const second = ctrls[1];
    expect(second).toBeDefined();
    await act(async () => {
      second.push(FINISH);
      second.push(DONE);
      second.close();
    });
  });
});

describe('accessibility', () => {
  it('has no axe violations on the empty screen, a conversation and the settings', async () => {
    mockFetch(() => sseResponse([delta('think', 'reasoning_content'), delta('# Title\n\n```js\nx()\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |'), FINISH, USAGE, DONE]));
    const { container } = renderApp();
    await screen.findByText('Model: qwen3.5-4b');
    expect(screen.getByRole('region', { name: 'Conversation' }).tabIndex).toBe(0);
    expect(await axeViolations(container)).toEqual([]);
    typeAndSend('hello');
    await screen.findByText('Title');
    await screen.findByLabelText('Reply statistics');
    expect(await axeViolations(container)).toEqual([]);
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }));
    expect(await axeViolations(container)).toEqual([]);
  });
});
