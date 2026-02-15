import {
  ActionIcon,
  AppShell,
  Badge,
  Box,
  Button,
  Card,
  Divider,
  Group,
  Loader,
  Modal,
  ScrollArea,
  Stack,
  Text,
  TextInput,
  Textarea,
  Title,
} from '@mantine/core';
import { Dropzone } from '@mantine/dropzone';
import { notifications } from '@mantine/notifications';
import { IconLink, IconPaperclip, IconRocket, IconSettings } from '@tabler/icons-react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import React from 'react';
import ReactMarkdown from 'react-markdown';

import { chat, createSession, getBackendUrl, listFiles, uploadFiles, type FileInfo } from '../lib/api';

function bytes(n: number) {
  const units = ['B', 'KB', 'MB', 'GB'];
  let v = n;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

export default function App() {
  const qc = useQueryClient();

  const [backendUrl, setBackendUrl] = React.useState(getBackendUrl());
  const [settingsOpen, setSettingsOpen] = React.useState(false);

  const [sessionId, setSessionId] = React.useState<string | null>(localStorage.getItem('sessionId'));
  const [message, setMessage] = React.useState('');
  const [chatLog, setChatLog] = React.useState<
    { role: 'user' | 'assistant'; content: string; citations?: any[] }[]
  >([]);

  const createSessionMut = useMutation({
    mutationFn: async () => {
      const res = await createSession();
      return res.session_id;
    },
    onSuccess: (id) => {
      setSessionId(id);
      localStorage.setItem('sessionId', id);
      notifications.show({ title: 'Session created', message: id });
    },
    onError: (e: any) => {
      notifications.show({ color: 'red', title: 'Failed to create session', message: String(e?.message || e) });
    },
  });

  const filesQuery = useQuery({
    queryKey: ['files', sessionId],
    queryFn: async () => (sessionId ? listFiles(sessionId) : []),
    enabled: !!sessionId,
    staleTime: 2000,
  });

  const uploadMut = useMutation({
    mutationFn: async (files: File[]) => {
      if (!sessionId) throw new Error('No session');
      return uploadFiles(sessionId, files);
    },
    onSuccess: (res) => {
      qc.invalidateQueries({ queryKey: ['files', sessionId] });
      const ok = res.results.filter((r) => r.ok).length;
      const bad = res.results.length - ok;
      notifications.show({
        title: 'Upload finished',
        message: `${ok} indexed${bad ? `, ${bad} skipped` : ''}`,
      });
    },
    onError: (e: any) => {
      notifications.show({ color: 'red', title: 'Upload failed', message: String(e?.message || e) });
    },
  });

  const chatMut = useMutation({
    mutationFn: async (text: string) => {
      if (!sessionId) throw new Error('No session');
      return chat(sessionId, text);
    },
    onMutate: async (text) => {
      setChatLog((l) => [...l, { role: 'user', content: text }]);
      setMessage('');
    },
    onSuccess: (res) => {
      setChatLog((l) => [...l, { role: 'assistant', content: res.answer, citations: res.citations }]);
    },
    onError: (e: any) => {
      notifications.show({ color: 'red', title: 'Chat failed', message: String(e?.message || e) });
    },
  });

  function saveBackendUrl() {
    localStorage.setItem('backendUrl', backendUrl);
    setSettingsOpen(false);
    notifications.show({ title: 'Saved', message: 'Backend URL updated' });
  }

  const hasSession = !!sessionId;

  return (
    <AppShell
      header={{ height: 64 }}
      navbar={{ width: 360, breakpoint: 'sm' }}
      padding="md"
    >
      <AppShell.Header>
        <Group h="100%" px="md" justify="space-between">
          <Group gap="sm">
            <motion.div initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
              <Title order={3}>RAG Engine</Title>
            </motion.div>
            <Badge variant="light" color={hasSession ? 'teal' : 'gray'}>
              {hasSession ? `Session: ${sessionId!.slice(0, 8)}…` : 'No session'}
            </Badge>
          </Group>
          <Group gap="xs">
            <ActionIcon variant="subtle" onClick={() => setSettingsOpen(true)} aria-label="Settings">
              <IconSettings size={18} />
            </ActionIcon>
          </Group>
        </Group>
      </AppShell.Header>

      <AppShell.Navbar p="md">
        <Stack gap="md" h="100%">
          <Card withBorder radius="md" p="md">
            <Group justify="space-between" mb="xs">
              <Text fw={700}>Workspace</Text>
              <Button
                leftSection={<IconRocket size={16} />}
                size="xs"
                variant="light"
                loading={createSessionMut.isPending}
                onClick={() => createSessionMut.mutate()}
              >
                New session
              </Button>
            </Group>
            <Text size="sm" c="dimmed">
              Upload files, then ask questions. The backend builds a RAG index (semantic if embeddings are available).
            </Text>
          </Card>

          <Card withBorder radius="md" p="md">
            <Group justify="space-between" mb="xs">
              <Group gap="xs">
                <IconPaperclip size={16} />
                <Text fw={700}>Upload</Text>
              </Group>
              <Badge variant="dot" color={filesQuery.data?.length ? 'teal' : 'gray'}>
                {filesQuery.data?.length || 0} files
              </Badge>
            </Group>

            <Dropzone
              disabled={!hasSession}
              onDrop={(accepted) => uploadMut.mutate(accepted)}
              maxSize={50 * 1024 * 1024}
              multiple
            >
              <Stack gap={6} p="md" align="center">
                <Text fw={700}>{hasSession ? 'Drop files here' : 'Create a session first'}</Text>
                <Text size="sm" c="dimmed">
                  PDF, DOCX, TXT, MD, PNG/JPG/WebP/GIF
                </Text>
              </Stack>
            </Dropzone>

            {uploadMut.isPending && (
              <Group mt="sm" gap="sm">
                <Loader size="sm" />
                <Text size="sm">Indexing...</Text>
              </Group>
            )}
          </Card>

          <Card withBorder radius="md" p="md" style={{ flex: 1, minHeight: 0 }}>
            <Group justify="space-between" mb="xs">
              <Text fw={700}>Files</Text>
              {filesQuery.isFetching ? <Loader size="sm" /> : null}
            </Group>
            <Divider mb="sm" />
            <ScrollArea h="100%" type="auto">
              <Stack gap="sm">
                {(filesQuery.data || []).map((f: FileInfo) => (
                  <Card key={f.id} withBorder radius="md" p="sm">
                    <Group justify="space-between" align="flex-start" gap="sm">
                      <Box style={{ minWidth: 0 }}>
                        <Text fw={700} lineClamp={1}>
                          {f.name}
                        </Text>
                        <Text size="xs" c="dimmed">
                          {f.mime} · {bytes(f.size_bytes)}
                        </Text>
                      </Box>
                      <Badge variant="light">{f.mime.startsWith('image/') ? 'Image' : 'Doc'}</Badge>
                    </Group>
                  </Card>
                ))}
                {!filesQuery.data?.length && (
                  <Text size="sm" c="dimmed">
                    No files yet.
                  </Text>
                )}
              </Stack>
            </ScrollArea>
          </Card>
        </Stack>
      </AppShell.Navbar>

      <AppShell.Main>
        <Stack gap="md" style={{ height: 'calc(100vh - 64px - 32px)' }}>
          <Card withBorder radius="md" p="md" style={{ flex: 1, minHeight: 0 }}>
            <Group justify="space-between" mb="xs">
              <Text fw={700}>Chat</Text>
              <Badge variant="light" color="gray">
                {backendUrl}
              </Badge>
            </Group>
            <Divider mb="sm" />

            <ScrollArea h="100%" type="auto">
              <Stack gap="md" pb="md">
                {chatLog.map((m, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.22 }}
                  >
                    <Card
                      withBorder
                      radius="md"
                      p="md"
                      style={{
                        background: m.role === 'user' ? 'var(--mantine-color-gray-0)' : 'white',
                        borderColor: m.role === 'user' ? 'var(--mantine-color-gray-3)' : 'var(--mantine-color-teal-2)',
                      }}
                    >
                      <Group justify="space-between" mb="xs">
                        <Badge color={m.role === 'user' ? 'gray' : 'teal'} variant="light">
                          {m.role}
                        </Badge>
                      </Group>
                      <ReactMarkdown>{m.content}</ReactMarkdown>

                      {m.role === 'assistant' && m.citations?.length ? (
                        <Box mt="md">
                          <Divider mb="sm" />
                          <Text size="sm" fw={700} mb={6}>
                            Sources
                          </Text>
                          <Stack gap="xs">
                            {m.citations.map((c: any) => (
                              <Card key={c.chunk_id} withBorder radius="md" p="sm">
                                <Group justify="space-between" align="flex-start" gap="sm">
                                  <Box style={{ minWidth: 0 }}>
                                    <Text size="sm" fw={700} lineClamp={1}>
                                      {c.file_name}
                                      {c.page ? ` · page ${c.page}` : ''}
                                    </Text>
                                    <Text size="xs" c="dimmed" lineClamp={2}>
                                      {c.snippet}
                                    </Text>
                                  </Box>
                                  <Badge variant="light">{Number(c.score).toFixed(3)}</Badge>
                                </Group>
                              </Card>
                            ))}
                          </Stack>
                        </Box>
                      ) : null}
                    </Card>
                  </motion.div>
                ))}

                {!chatLog.length && (
                  <Text c="dimmed">
                    Upload a PDF/doc first, then ask something like: "Summarize the key points" or "What is the refund policy?"
                  </Text>
                )}
              </Stack>
            </ScrollArea>
          </Card>

          <Card withBorder radius="md" p="md">
            <Group align="flex-end" gap="sm">
              <Textarea
                value={message}
                onChange={(e) => setMessage(e.currentTarget.value)}
                placeholder={hasSession ? 'Ask a question...' : 'Create a session first...'}
                autosize
                minRows={2}
                style={{ flex: 1 }}
                disabled={!hasSession}
              />
              <Button
                onClick={() => message.trim() && chatMut.mutate(message.trim())}
                loading={chatMut.isPending}
                disabled={!hasSession}
              >
                Ask
              </Button>
            </Group>
          </Card>
        </Stack>
      </AppShell.Main>

      <Modal opened={settingsOpen} onClose={() => setSettingsOpen(false)} title="Settings" centered>
        <Stack>
          <Text size="sm" c="dimmed">
            Backend URL is stored in your browser localStorage.
          </Text>
          <TextInput
            leftSection={<IconLink size={16} />}
            label="Backend URL"
            value={backendUrl}
            onChange={(e) => setBackendUrl(e.currentTarget.value)}
            placeholder="http://localhost:8000"
          />
          <Group justify="flex-end">
            <Button variant="light" onClick={() => setSettingsOpen(false)}>
              Cancel
            </Button>
            <Button onClick={saveBackendUrl}>Save</Button>
          </Group>
        </Stack>
      </Modal>
    </AppShell>
  );
}
