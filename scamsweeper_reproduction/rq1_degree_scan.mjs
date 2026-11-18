// Full static degree scan for the released ScamSweeper RQ1 NetworkX pickles.
// It starts at the previously verified _node/_adj boundary and parses only
// primitive pickle opcodes. No pickle globals, reducers, or constructors run.

import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { Reader, opcode } from '../tools/rq1_pickle_primitives.mjs';

const NUM = Object.freeze({ kind: 'number' });
const MARK = Object.freeze({ kind: 'mark' });
const OPAQUE = Object.freeze({ kind: 'opaque' });

class TextToken {
  constructor(value) { this.kind = 'text'; this.value = value; this.memo = null; }
}

class RefToken {
  constructor(index) { this.kind = 'ref'; this.index = index; }
}

class Frame {
  constructor() { this.tag = 'empty'; this.edges = 0; }
}

class PlainDict {
  constructor() { this.kind = 'dict'; }
}

function textValue(value) {
  return value instanceof TextToken ? value.value : null;
}

function memoIndex(value) {
  if (value instanceof RefToken) return value.index;
  if (value instanceof TextToken && Number.isInteger(value.memo)) return value.memo;
  throw new Error(`Graph key was not memoized: ${JSON.stringify(value)}`);
}

function increment(array, index, amount) {
  const next = array[index] + amount;
  if (!Number.isSafeInteger(next) || next > 0xffffffff) {
    throw new Error(`Degree overflow at memo ${index}: ${next}`);
  }
  array[index] = next;
}

function addHistogram(histogram, degree, count = 1) {
  histogram.set(degree, (histogram.get(degree) ?? 0) + count);
}

function moments(histogram, nodeCount) {
  let sum = 0n;
  let sumSquares = 0n;
  let maximum = 0;
  let counted = 0;
  for (const [degree, frequency] of histogram) {
    const d = BigInt(degree);
    const f = BigInt(frequency);
    sum += d * f;
    sumSquares += d * d * f;
    counted += frequency;
    maximum = Math.max(maximum, degree);
  }
  if (counted !== nodeCount) throw new Error(`Histogram count ${counted} != ${nodeCount}`);
  const n = Number(nodeCount);
  const sumNumber = Number(sum);
  const sumSquaresNumber = Number(sumSquares);
  const mean = sumNumber / n;
  const populationVariance = Math.max(0, sumSquaresNumber / n - mean * mean);
  const sampleVariance = n > 1 ? populationVariance * n / (n - 1) : 0;
  return {
    count: nodeCount,
    sum_exact: sum.toString(),
    sum_squares_exact: sumSquares.toString(),
    mean,
    maximum,
    population_variance: populationVariance,
    population_sd: Math.sqrt(populationVariance),
    sample_variance: sampleVariance,
    sample_sd: Math.sqrt(sampleVariance),
  };
}

function writeHistogram(outputPath, inHist, outHist, totalHist) {
  const degrees = [...new Set([...inHist.keys(), ...outHist.keys(), ...totalHist.keys()])]
    .sort((a, b) => a - b);
  const rows = ['degree,in_frequency,out_frequency,total_frequency'];
  for (const degree of degrees) {
    rows.push(`${degree},${inHist.get(degree) ?? 0},${outHist.get(degree) ?? 0},${totalHist.get(degree) ?? 0}`);
  }
  fs.writeFileSync(outputPath, rows.join('\n') + '\n');
}

function scanNodePrefix(reader, boundary, name) {
  const UNSET = 0xffffffff;
  const memoToNode = new Uint32Array(boundary.next_memo_index_at_boundary);
  memoToNode.fill(UNSET);
  const stringToNode = new Map();
  let nextMemo = 0;
  let last = OPAQUE;
  let waiting = false;
  let root = null;
  const stack = [];
  let nodeCount = 0;

  function saveMemo(index, value) {
    if (value instanceof TextToken) value.memo = index;
  }

  function recordRootKey(key) {
    const memo = memoIndex(key);
    if (memo >= memoToNode.length) throw new Error(`${name}: node-key memo exceeds verified boundary`);
    if (memoToNode[memo] !== UNSET) throw new Error(`${name}: duplicate node-key memo ${memo}`);
    const compact = nodeCount++;
    memoToNode[memo] = compact;
    if (key instanceof TextToken) {
      if (stringToNode.has(key.value)) throw new Error(`${name}: duplicate node identifier ${key.value}`);
      stringToNode.set(key.value, compact);
    }
  }

  while (reader.pos < boundary.last_opcode_position) {
    const item = opcode(reader);
    if (item.op === 'HEADER') continue;
    if (item.op === 'MEMO') {
      const index = item.index ?? nextMemo;
      saveMemo(index, root ? stack.at(-1) : last);
      nextMemo = Math.max(nextMemo, index + 1);
      continue;
    }
    if (!root) {
      if (item.op === 'STRING') {
        last = new TextToken(item.v);
        if (item.v === '_node') waiting = true;
      } else if (waiting && item.op === 'DICT') {
        root = new PlainDict();
        stack.push(root);
        waiting = false;
      } else {
        last = OPAQUE;
      }
      continue;
    }
    if (item.op === 'STRING') stack.push(new TextToken(item.v));
    else if (item.op === 'NUMBER' || item.op === 'OPAQUE') stack.push(OPAQUE);
    else if (item.op === 'GET') stack.push(new RefToken(item.index));
    else if (item.op === 'DICT') stack.push(new PlainDict());
    else if (item.op === 'MARK') stack.push(MARK);
    else if (item.op === 'SET') {
      const value = stack.pop();
      const key = stack.pop();
      const parent = stack.at(-1);
      if (parent === root) recordRootKey(key);
      else if (!(parent instanceof PlainDict)) throw new Error(`${name}: invalid node attribute dictionary`);
    } else if (item.op === 'SETS') {
      let marker = stack.length - 1;
      while (marker >= 0 && stack[marker] !== MARK) marker--;
      if (marker < 1 || (stack.length - marker - 1) % 2) throw new Error(`${name}: malformed node dictionary batch`);
      const parent = stack[marker - 1];
      for (let index = marker + 1; index < stack.length; index += 2) {
        if (parent === root) recordRootKey(stack[index]);
        else if (!(parent instanceof PlainDict)) throw new Error(`${name}: invalid node attribute batch`);
      }
      stack.length = marker;
    } else {
      throw new Error(`${name}: unsupported ${item.op} in node table at ${item.pos}`);
    }
  }
  if (reader.pos !== boundary.last_opcode_position) throw new Error(`${name}: missed verified node boundary`);
  if (nodeCount !== boundary.sampled_nodes) throw new Error(`${name}: node prefix count ${nodeCount} != ${boundary.sampled_nodes}`);
  if (nextMemo !== boundary.next_memo_index_at_boundary) throw new Error(`${name}: memo boundary changed`);
  console.log(`${name}: ${nodeCount.toLocaleString()} node keys indexed`);
  return { memoToNode, stringToNode, unset: UNSET, nextMemo };
}

function scanGraph(name, boundary, outputDir) {
  if (!boundary.node_table_complete) throw new Error(`${name}: node boundary is not complete`);
  const before = fs.statSync(boundary.file);
  if (before.size !== boundary.file_bytes) throw new Error(`${name}: raw file size changed`);

  const memoLimit = boundary.next_memo_index_at_boundary;
  if (!Number.isInteger(memoLimit) || memoLimit < 1) throw new Error(`${name}: invalid memo boundary`);
  // Memo IDs also cover every edge-attribute object and therefore grow much
  // faster than the node population. Map only node-key memo IDs to compact
  // array positions so memory scales with nodes rather than pickle objects.
  const inDegree = new Uint32Array(boundary.sampled_nodes);
  const outDegree = new Uint32Array(boundary.sampled_nodes);
  const sourceSeen = new Uint8Array(boundary.sampled_nodes);
  const reader = new Reader(boundary.file);
  const prefix = scanNodePrefix(reader, boundary, name);
  const adjacencyMemoToNode = new Map();
  const pendingIncoming = new Map();
  let indexedNodes = boundary.sampled_nodes;

  let nextMemo = prefix.nextMemo;
  let last = OPAQUE;
  let root = null;
  const stack = [];
  let complete = false;
  let edgeRecords = 0;
  let sourceBlocks = 0;
  let targetPairBlocks = 0;
  let lastProgress = 0;
  const started = Date.now();

  function saveMemo(index, value) {
    if (value instanceof TextToken) {
      value.memo = index;
      if (prefix.stringToNode.has(value.value)) {
        adjacencyMemoToNode.set(index, prefix.stringToNode.get(value.value));
      }
    }
  }

  function compactNodeIndex(key, role) {
    const memo = memoIndex(key);
    if (key instanceof TextToken && prefix.stringToNode.has(key.value)) {
      const compact = prefix.stringToNode.get(key.value);
      adjacencyMemoToNode.set(memo, compact);
      return compact;
    }
    if (memo < prefix.memoToNode.length && prefix.memoToNode[memo] !== prefix.unset) {
      return prefix.memoToNode[memo];
    }
    if (adjacencyMemoToNode.has(memo)) return adjacencyMemoToNode.get(memo);
    if (role !== 'source') return null;
    throw new Error(`${name}: source memo ${memo} has no canonical node identifier mapping`);
  }

  function assign(parent, key, value) {
    if (!(parent instanceof Frame)) throw new Error(`${name}: invalid dictionary assignment`);
    if (parent === root) {
      if (!(value instanceof Frame) || !['neighbors', 'empty'].includes(value.tag)) {
        throw new Error(`${name}: invalid root adjacency value`);
      }
      const sourceNode = compactNodeIndex(key, 'source');
      const sourceMemo = memoIndex(key);
      adjacencyMemoToNode.set(sourceMemo, sourceNode);
      if (sourceSeen[sourceNode]) throw new Error(`${name}: duplicate source adjacency block for node ${sourceNode}`);
      sourceSeen[sourceNode] = 1;
      if (pendingIncoming.has(sourceMemo)) {
        increment(inDegree, sourceNode, pendingIncoming.get(sourceMemo));
        pendingIncoming.delete(sourceMemo);
      }
      increment(outDegree, sourceNode, value.edges);
      sourceBlocks++;
      return;
    }
    if (value instanceof Frame) {
      if (parent.tag === 'leaf') throw new Error(`${name}: nested edge attributes`);
      if (value.tag === 'leaf' || (value.tag === 'empty' && key === NUM)) {
        if (!['empty', 'pair'].includes(parent.tag)) throw new Error(`${name}: mixed edge dictionary`);
        parent.tag = 'pair';
        parent.edges++;
        edgeRecords++;
      } else if (['pair', 'empty'].includes(value.tag)) {
        if (!['empty', 'neighbors'].includes(parent.tag)) throw new Error(`${name}: invalid adjacency depth`);
        parent.tag = 'neighbors';
        parent.edges += value.edges;
        const targetMemo = memoIndex(key);
        const targetNode = compactNodeIndex(key, 'target');
        if (targetNode === null) {
          pendingIncoming.set(targetMemo, (pendingIncoming.get(targetMemo) ?? 0) + value.edges);
        } else {
          increment(inDegree, targetNode, value.edges);
        }
        targetPairBlocks++;
      } else {
        throw new Error(`${name}: unsupported nested frame`);
      }
    } else {
      if (!['empty', 'leaf'].includes(parent.tag)) throw new Error(`${name}: non-flat attributes`);
      parent.tag = 'leaf';
    }
  }

  try {
    for (;;) {
      const item = opcode(reader);
      if (item.op === 'HEADER') continue;
      if (item.op === 'MEMO') {
        const index = item.index ?? nextMemo;
        saveMemo(index, root ? stack.at(-1) : last);
        nextMemo = Math.max(nextMemo, index + 1);
        continue;
      }
      if (!root) {
        if (item.op === 'STRING') {
          last = new TextToken(item.v);
          continue;
        }
        if (item.op === 'DICT') {
          if (textValue(last) !== '_adj') throw new Error(`${name}: expected _adj before root dictionary`);
          root = new Frame();
          stack.push(root);
          continue;
        }
        throw new Error(`${name}: invalid adjacency entry at ${item.pos}`);
      }
      if (item.op === 'STRING') {
        if (stack.length === 1 && ['_succ', '_pred', 'adj'].includes(item.v)) {
          complete = true;
          break;
        }
        stack.push(new TextToken(item.v));
      } else if (item.op === 'NUMBER') {
        stack.push(NUM);
      } else if (item.op === 'GET') {
        stack.push(new RefToken(item.index));
      } else if (item.op === 'DICT') {
        stack.push(new Frame());
      } else if (item.op === 'MARK') {
        stack.push(MARK);
      } else if (item.op === 'SET') {
        if (stack.length === 1) { complete = true; break; }
        const value = stack.pop();
        const key = stack.pop();
        assign(stack.at(-1), key, value);
      } else if (item.op === 'SETS') {
        let marker = stack.length - 1;
        while (marker >= 0 && stack[marker] !== MARK) marker--;
        if (marker < 1 || (stack.length - marker - 1) % 2) throw new Error(`${name}: malformed dictionary batch`);
        const parent = stack[marker - 1];
        for (let index = marker + 1; index < stack.length; index += 2) {
          assign(parent, stack[index], stack[index + 1]);
        }
        stack.length = marker;
      } else {
        throw new Error(`${name}: unsupported ${item.op} inside adjacency at ${item.pos}`);
      }
      if (edgeRecords - lastProgress >= 10_000_000) {
        lastProgress = edgeRecords;
        console.log(`${name}: ${edgeRecords.toLocaleString()} edge records scanned`);
      }
    }
  } finally {
    reader.close();
  }

  if (!complete) throw new Error(`${name}: adjacency scan did not complete`);
  if (pendingIncoming.size) {
    const examples = [...pendingIncoming.keys()].slice(0, 10);
    throw new Error(`${name}: ${pendingIncoming.size} target memos never resolved to source nodes: ${examples.join(',')}`);
  }
  const after = fs.statSync(boundary.file);
  if (after.size !== before.size || after.mtimeMs !== before.mtimeMs) throw new Error(`${name}: raw file changed during scan`);
  if (edgeRecords <= 0 || sourceBlocks <= 0) throw new Error(`${name}: empty scan result`);

  const inHist = new Map();
  const outHist = new Map();
  const totalHist = new Map();
  const sourceInHist = new Map();
  const sourceOutHist = new Map();
  const sourceTotalHist = new Map();
  for (let index = 0; index < indexedNodes; index++) {
    const incoming = inDegree[index];
    const outgoing = outDegree[index];
    addHistogram(inHist, incoming);
    addHistogram(outHist, outgoing);
    addHistogram(totalHist, incoming + outgoing);
    if (sourceSeen[index]) {
      addHistogram(sourceInHist, incoming);
      addHistogram(sourceOutHist, outgoing);
      addHistogram(sourceTotalHist, incoming + outgoing);
    }
  }
  const implicitZeroNodes = 0;

  fs.mkdirSync(outputDir, { recursive: true });
  writeHistogram(path.join(outputDir, `${name}_degree_frequency.csv`), inHist, outHist, totalHist);
  const report = {
    graph: name,
    complete: true,
    source_file: boundary.file,
    source_file_bytes: before.size,
    node_count: boundary.sampled_nodes,
    indexed_node_memos: indexedNodes,
    implicit_zero_degree_nodes: implicitZeroNodes,
    source_blocks: sourceBlocks,
    target_pair_blocks: targetPairBlocks,
    edge_records: edgeRecords,
    degrees_all_node_records: {
      in: moments(inHist, boundary.sampled_nodes),
      out: moments(outHist, boundary.sampled_nodes),
      total: moments(totalHist, boundary.sampled_nodes),
    },
    degrees_adjacency_sources: {
      in: moments(sourceInHist, sourceBlocks),
      out: moments(sourceOutHist, sourceBlocks),
      total: moments(sourceTotalHist, sourceBlocks),
    },
    elapsed_seconds: (Date.now() - started) / 1000,
    pickle_objects_loaded: false,
    model_training_performed: false,
  };
  fs.writeFileSync(path.join(outputDir, `${name}_degree_report.json`), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify({
    graph: name,
    nodes: report.node_count,
    adjacency_sources: sourceBlocks,
    edges: edgeRecords,
    paper_scope_total_sd: report.degrees_adjacency_sources.total.population_sd,
    all_node_records_total_sd: report.degrees_all_node_records.total.population_sd,
  }));
  return report;
}

function parseArgs(argv) {
  const args = {};
  for (let index = 2; index < argv.length; index += 2) args[argv[index]] = argv[index + 1];
  return args;
}

const args = parseArgs(process.argv);
if (!args['--boundaries'] || !args['--output-dir']) {
  throw new Error('Required: --boundaries PATH --output-dir PATH [--graphs normal,phish,scams]');
}
const boundariesPath = path.resolve(args['--boundaries']);
const outputDir = path.resolve(args['--output-dir']);
if (fs.existsSync(outputDir)) throw new Error('Output directory already exists; choose a new path');
fs.mkdirSync(outputDir, { recursive: true });
const boundaries = JSON.parse(fs.readFileSync(boundariesPath, 'utf8'));
const graphNames = (args['--graphs'] ?? 'normal,phish,scams').split(',');
const reports = [];
for (const name of graphNames) {
  if (!boundaries[name]) throw new Error(`Unknown graph: ${name}`);
  reports.push(scanGraph(name, boundaries[name], outputDir));
}
const summary = {
  status: 'complete',
  graphs: reports,
  inputs: {
    boundaries: boundariesPath,
    boundaries_sha256: crypto.createHash('sha256').update(fs.readFileSync(boundariesPath)).digest('hex'),
  },
  code_sha256: crypto.createHash('sha256').update(fs.readFileSync(new URL(import.meta.url))).digest('hex'),
  pickle_objects_loaded: false,
  model_training_performed: false,
};
fs.writeFileSync(path.join(outputDir, 'summary.json'), JSON.stringify(summary, null, 2) + '\n');

