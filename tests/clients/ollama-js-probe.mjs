import assert from 'node:assert/strict'
import { Ollama } from 'ollama'

const host = process.env.CLOZN_OLLAMA_HOST
assert.ok(host, 'CLOZN_OLLAMA_HOST must point at the test gateway')

const model = 'sdk-fixture'
const client = new Ollama({ host })

const listed = await client.list()
assert.ok(Array.isArray(listed.models), 'list() should return a models array')
assert.ok(listed.models.some(item => item.name === model), `${model} should be listed by /api/tags`)

const chat = await client.chat({
  model,
  messages: [{ role: 'user', content: 'nonstream chat' }],
  stream: false,
})
assert.equal(chat.message.role, 'assistant')
assert.equal(chat.message.content, 'SDK nonstream chat.')
assert.equal(chat.done, true)
assert.match(chat.clozn_run_id, /^run_/)

const generated = await client.generate({
  model,
  prompt: 'nonstream generate',
  stream: false,
})
assert.equal(generated.response, 'SDK nonstream generate.')
assert.equal(generated.done, true)
assert.match(generated.clozn_run_id, /^run_/)

const chatStream = await client.chat({
  model,
  messages: [{ role: 'user', content: 'stream chat' }],
  stream: true,
})
const chatPieces = []
let chatTerminal
for await (const part of chatStream) {
  if (part.message?.content) chatPieces.push(part.message.content)
  if (part.done) chatTerminal = part
}
assert.equal(chatPieces.join(''), 'SDK stream chat.')
assert.ok(chatTerminal, 'streaming chat should expose the terminal NDJSON object')
assert.match(chatTerminal.clozn_run_id, /^run_/)

const generateStream = await client.generate({
  model,
  prompt: 'stream generate',
  stream: true,
})
const generatePieces = []
let generateTerminal
for await (const part of generateStream) {
  if (part.response) generatePieces.push(part.response)
  if (part.done) generateTerminal = part
}
assert.equal(generatePieces.join(''), 'SDK stream generate.')
assert.ok(generateTerminal, 'streaming generate should expose the terminal NDJSON object')
assert.match(generateTerminal.clozn_run_id, /^run_/)

process.stdout.write(JSON.stringify({
  modelListed: true,
  nonstreamChatRunId: chat.clozn_run_id,
  nonstreamGenerateRunId: generated.clozn_run_id,
  streamChatRunId: chatTerminal.clozn_run_id,
  streamGenerateRunId: generateTerminal.clozn_run_id,
}))

