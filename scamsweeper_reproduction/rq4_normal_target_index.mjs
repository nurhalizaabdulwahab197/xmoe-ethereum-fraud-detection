// Static pickle indexer. No pickle globals, constructors or reducers are run.
// Indexes committed outgoing adjacency dictionaries and samples counterparty
// blocks for fixed target addresses. All original files are opened read-only.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

const NUM = Object.freeze({kind:'number'});
const OBJ = Object.freeze({kind:'opaque'});
class Reader {
  constructor(file) { this.fd=fs.openSync(file,'r'); this.buf=Buffer.allocUnsafe(8*1024*1024); this.i=0; this.n=0; this.pos=0; }
  fill(){this.n=fs.readSync(this.fd,this.buf,0,this.buf.length,null);this.i=0;if(!this.n)throw Error('Unexpected EOF');}
  byte(){if(this.i===this.n)this.fill();this.pos++;return this.buf[this.i++];}
  bytes(n){if(n<0||!Number.isSafeInteger(n)||n>16*1024*1024)throw Error('Oversized scalar');if(this.n-this.i>=n){const b=this.buf.subarray(this.i,this.i+n);this.i+=n;this.pos+=n;return b;}const b=Buffer.allocUnsafe(n);for(let i=0;i<n;i++)b[i]=this.byte();return b;}
  skip(n){while(n){if(this.i===this.n)this.fill();const take=Math.min(n,this.n-this.i);this.i+=take;this.pos+=take;n-=take;}}
  u32(){return this.bytes(4).readUInt32LE(0);}
  u64(){const n=Number(this.bytes(8).readBigUInt64LE(0));if(!Number.isSafeInteger(n))throw Error('Unsafe length');return n;}
  line(){const a=[];for(let i=0;i<1048576;i++){const b=this.byte();if(b===10)return Buffer.from(a).toString('utf8');a.push(b);}throw Error('Oversized line');}
  close(){fs.closeSync(this.fd);}
}
function opcode(r){const pos=r.pos,c=r.byte();switch(c){
  case 0x80:r.byte();return {op:'HEADER',pos};case 0x95:r.skip(8);return {op:'HEADER',pos};
  case 0x8c:return {op:'STRING',v:r.bytes(r.byte()).toString('utf8'),pos};
  case 0x58:return {op:'STRING',v:r.bytes(r.u32()).toString('utf8'),pos};
  case 0x8d:return {op:'STRING',v:r.bytes(r.u64()).toString('utf8'),pos};
  case 0x56:return {op:'STRING',v:r.line(),pos};
  case 0x94:return {op:'MEMO',pos};case 0x71:return {op:'MEMO',index:r.byte(),pos};case 0x72:return {op:'MEMO',index:r.u32(),pos};case 0x70:return {op:'MEMO',index:Number(r.line()),pos};
  case 0x68:return {op:'GET',index:r.byte(),pos};case 0x6a:return {op:'GET',index:r.u32(),pos};case 0x67:return {op:'GET',index:Number(r.line()),pos};
  case 0x7d:return {op:'DICT',pos};case 0x28:return {op:'MARK',pos};case 0x73:return {op:'SET',pos};case 0x75:return {op:'SETS',pos};case 0x2e:return {op:'STOP',pos};
  case 0x4a:r.skip(4);return {op:'NUMBER',pos};case 0x4b:r.skip(1);return {op:'NUMBER',pos};case 0x4d:r.skip(2);return {op:'NUMBER',pos};case 0x47:r.skip(8);return {op:'NUMBER',pos};
  case 0x8a:r.skip(r.byte());return {op:'NUMBER',pos};case 0x8b:r.skip(r.u32());return {op:'NUMBER',pos};
  case 0x49:case 0x4c:case 0x46:r.line();return {op:'NUMBER',pos};case 0x4e:case 0x88:case 0x89:return {op:'NUMBER',pos};
  case 0x63:r.line();r.line();return {op:'OPAQUE',pos};
  case 0x82:r.skip(1);return {op:'OPAQUE',pos};case 0x83:r.skip(2);return {op:'OPAQUE',pos};case 0x84:r.skip(4);return {op:'OPAQUE',pos};
  case 0x81:case 0x85:case 0x86:case 0x87:case 0x93:case 0x92:case 0x62:case 0x52:case 0x29:case 0x74:case 0x5d:case 0x6c:case 0x61:case 0x65:case 0x30:case 0x31:case 0x32:case 0x8f:case 0x90:case 0x91:return {op:'OPAQUE',pos};
  case 0x42:r.skip(r.u32());return {op:'OPAQUE',pos};case 0x43:r.skip(r.byte());return {op:'OPAQUE',pos};case 0x8e:case 0x96:r.skip(r.u64());return {op:'OPAQUE',pos};
  default:throw Error(`Unsupported opcode 0x${c.toString(16)} at ${pos}`);
}}
class Bits {
 constructor(){this.b=Buffer.alloc(4*1024*1024);}
 set(i){if(!Number.isInteger(i)||i<0||i>=2**32)throw Error('Memo ID out of range');const k=Math.floor(i/8);if(k>=this.b.length){let n=this.b.length;while(n<=k)n*=2;const v=Buffer.alloc(n);this.b.copy(v);this.b=v;}this.b[k]|=1<<(i%8);}
 has(i){const k=Math.floor(i/8);return k<this.b.length&&Boolean(this.b[k]&(1<<(i%8)));}
}
class BufferedFile {
 constructor(file){this.fd=fs.openSync(file,'wx');this.b=Buffer.allocUnsafe(1024*1024);this.n=0;this.offset=0;}
 write(value){let i=0;while(i<value.length){const take=Math.min(this.b.length-this.n,value.length-i);value.copy(this.b,this.n,i,i+take);this.n+=take;this.offset+=take;i+=take;if(this.n===this.b.length)this.flush();}}
 flush(){let written=0;while(written<this.n)written+=fs.writeSync(this.fd,this.b,written,this.n-written);this.n=0;}
 close(){this.flush();fs.closeSync(this.fd);}
}
class Strings {
 constructor(folder,enabled){this.enabled=enabled;this.count=0;this.aliasCount=0;if(enabled){this.index=new BufferedFile(path.join(folder,'strings.idx'));this.data=new BufferedFile(path.join(folder,'strings.dat'));this.aliases=new BufferedFile(path.join(folder,'strings.aliases'));}}
 add(i,s){if(!this.enabled)return;const raw=Buffer.from(s,'utf8'), entry=Buffer.allocUnsafe(16);entry.writeUInt32LE(i,0);entry.writeUInt32LE(raw.length,4);entry.writeBigUInt64LE(BigInt(this.data.offset),8);this.index.write(entry);this.data.write(raw);this.count++;}
 alias(i,j){if(!this.enabled)return;const b=Buffer.allocUnsafe(8);b.writeUInt32LE(i,0);b.writeUInt32LE(j,4);this.aliases.write(b);this.aliasCount++;}
 close(){if(this.enabled){this.index.close();this.data.close();this.aliases.close();}}
}
function priority(offset,seed){let x=((offset>>>0)^Math.floor(offset/2**32)^seed)>>>0;x=Math.imul(x^(x>>>16),0x7feb352d);x=Math.imul(x^(x>>>15),0x846ca68b);return (x^(x>>>16))>>>0;}
function less(a,b){return a.priority<b.priority||(a.priority===b.priority&&a.start<b.start);}
class BottomK {
 constructor(k){this.k=k;this.a=[];}
 offer(x){const a=this.a;if(a.length<this.k){a.push(x);let i=a.length-1;while(i){const p=(i-1)>>1;if(!less(a[p],a[i]))break;[a[p],a[i]]=[a[i],a[p]];i=p;}}else if(less(x,a[0])){a[0]=x;let i=0;for(;;){let j=i,l=i*2+1,r=l+1;if(l<a.length&&less(a[j],a[l]))j=l;if(r<a.length&&less(a[j],a[r]))j=r;if(j===i)break;[a[j],a[i]]=[a[i],a[j]];i=j;}}}
 sorted(){return [...this.a].sort((a,b)=>a.start-b.start);}
}
class Frame {
 constructor(start,memo,view){this.start=start;this.end=start+1;this.memo=memo;this.view=view;this.tag='empty';this.edges=0;this.incoming=null;this.outgoing=null;this.sourceTarget=null;}
}
function ref(i){return {kind:'ref',index:i};}
function keyJSON(v){if(typeof v==='string')return {kind:'literal',value:v};if(v?.kind==='ref')return {kind:'memo',index:v.index};throw Error('Nonstring source/target key');}
const KEEP=new Set(['amount','timestamp','value','time','hash','txhash','tx_hash','transactionHash','transaction_hash','transaction_id','txid','blockNumber','block_number','fee','gas','gasPrice','gas_price','nonce']);

function indexGraph(group,boundary,targets,folder,options){
 fs.mkdirSync(folder);const startTime=Date.now(), stat=fs.statSync(boundary.file);
 if(stat.size!==boundary.file_bytes)throw Error('Input file size changed');
 const r=new Reader(boundary.file), bits=new Bits(), strings=new Strings(folder,options.log);
 const targetSet=new Set(targets.map(x=>x.address.toLowerCase()));const selectedMemo=new Map();let nextMemo=0,last=OBJ;
 let meta=Object.assign(Object.create(null),boundary.scalar_memo_at_boundary);let root=null,stack=[],nodesDone=false,complete=false;
 let roots=0,edges=0,lastLog=0,stop='preflight limit',adjStart=boundary.last_opcode_position;
 const outgoing=new Map(),incoming=new Map(),sourceTargets=new Set();
 function selected(v){const s=typeof v==='string'?v:v?.kind==='ref'?selectedMemo.get(v.index):undefined;return typeof s==='string'&&targetSet.has(s.toLowerCase())?s.toLowerCase():null;}
 function memoGet(i){if(bits.has(i))throw Error(`Opaque/dictionary memo reference ${i} inside outgoing adjacency`);if(selectedMemo.has(i))return selectedMemo.get(i);if(Object.hasOwn(meta,i))return meta[i];return ref(i);}
 function saveMemo(i,v){if(i>=2**32)throw Error('Too many memo objects');if(v instanceof Frame||v===OBJ){bits.set(i);if(v instanceof Frame)v.end=r.pos;}else if(typeof v==='string'){strings.add(i,v);if(targetSet.has(v.toLowerCase()))selectedMemo.set(i,v);if(nodesDone&&(KEEP.has(v)||targetSet.has(v.toLowerCase()))){meta=Object.assign(Object.create(null),meta,{[i]:v});}}else if(v?.kind==='ref')strings.alias(i,v.index);}
 function offer(map,target,item){if(!map.has(target))map.set(target,options.all?[]:new BottomK(options.k));const bucket=map.get(target);if(options.all)bucket.push(item);else bucket.offer(item);}
 function block(f,target,key){return {target,start:f.start,end:f.end,memo_start:f.memo,scalar_snapshot:options.all?null:f.view,edge_records:f.edges,peer:keyJSON(key),priority:priority(f.start,options.seed)};}
 function assign(parent,key,value){
  if(!(parent instanceof Frame))throw Error('Invalid dictionary assignment');parent.end=r.pos;
  if(parent===root){
   roots++;if(!(value instanceof Frame)||!['neighbors','empty'].includes(value.tag))throw Error('Root adjacency value has invalid dictionary depth');
   const target=selected(key);if(target){sourceTargets.add(target);if(value.outgoing){const blocks=options.all?value.outgoing:value.outgoing.sorted();for(const b of blocks)offer(outgoing,target,{...b,direction:'out',source:keyJSON(key)});}}
   if(value.incoming)for(const b of value.incoming)offer(incoming,b.target,{...b,direction:'in',source:keyJSON(key)});
   return;
  }
  if(value instanceof Frame){
   if(parent.tag==='leaf')throw Error('Nested edge attributes need a schema extension');
   if(value.tag==='leaf'||(value.tag==='empty'&&key===NUM)){if(parent.tag!=='empty'&&parent.tag!=='pair')throw Error('Mixed dictionary shape');parent.tag='pair';parent.edges++;edges++;}
   else if(value.tag==='pair'||value.tag==='empty'){
    if(parent.tag!=='empty'&&parent.tag!=='neighbors')throw Error('Unexpected adjacency depth');parent.tag='neighbors';parent.edges+=value.edges;
    if(value.edges){const target=selected(key);if(target){parent.incoming??=[];parent.incoming.push(block(value,target,key));}if(!options.all||parent.sourceTarget){parent.outgoing??=options.all?[]:new BottomK(options.k);if(options.all)parent.outgoing.push(block(value,null,key));else parent.outgoing.offer(block(value,null,key));}}
   }else throw Error('Unexpected nested dictionary');
  }else{if(parent.tag!=='empty'&&parent.tag!=='leaf')throw Error('Non-flat edge attributes');parent.tag='leaf';}
 }
 try{
  for(;;){
   if(!nodesDone&&r.pos===adjStart){if(nextMemo!==boundary.next_memo_index_at_boundary)throw Error('Node memo count differs from recorded boundary');nodesDone=true;for(const [i,v]of selectedMemo)meta[i]=v;console.log(`${group}: node memo index complete (${nextMemo.toLocaleString()} slots)`);}
   if(!nodesDone&&r.pos>adjStart)throw Error('Missed node boundary');
   if(nodesDone&&options.max>0&&r.pos-adjStart>=options.max){stop='preflight byte limit';break;}
   const o=opcode(r);if(o.op==='HEADER')continue;
   if(o.op==='MEMO'){const i=o.index??nextMemo;saveMemo(i,nodesDone&&root?stack.at(-1):last);nextMemo=Math.max(nextMemo,i+1);continue;}
   if(!nodesDone){if(o.op==='STRING')last=o.v;else if(o.op==='NUMBER')last=NUM;else if(o.op==='GET')last=bits.has(o.index)?OBJ:selectedMemo.get(o.index)??ref(o.index);else last=OBJ;continue;}
   if(!root){if(o.op==='STRING'&&o.v==='_adj'){last=o.v;continue;}if(o.op==='DICT'){root=new Frame(o.pos,nextMemo,meta);stack.push(root);continue;}throw Error('Invalid adjacency entry point');}
   if(o.op==='STRING'){
    if(stack.length===1&&['_succ','_pred','adj'].includes(o.v)){complete=true;stop='next graph field';break;}stack.push(o.v);
   }else if(o.op==='NUMBER')stack.push(NUM);
   else if(o.op==='GET')stack.push(memoGet(o.index));
   else if(o.op==='DICT'){const frame=new Frame(o.pos,nextMemo,meta);if(options.all)frame.sourceTarget=selected(stack.at(-1));stack.push(frame);}
   else if(o.op==='MARK')stack.push({kind:'mark'});
   else if(o.op==='SET'||o.op==='SETS'){
    if(o.op==='SET'&&stack.length===1){complete=true;stop='outer graph assignment';break;}
    if(o.op==='SET'){const v=stack.pop(),k=stack.pop();assign(stack.at(-1),k,v);}
    else{let j=stack.length-1;while(j>=0&&stack[j]?.kind!=='mark')j--;if(j<1||(stack.length-j-1)%2)throw Error('Invalid dictionary batch');const p=stack[j-1];for(let i=j+1;i<stack.length;i+=2)assign(p,stack[i],stack[i+1]);stack.length=j;}
   }else throw Error(`Unsupported ${o.op} inside adjacency at ${o.pos}`);
   if(edges-lastLog>=1000000){lastLog=edges;console.log(`${group}: ${edges.toLocaleString()} edge records, ${roots.toLocaleString()} committed source blocks, ${((r.pos-adjStart)/1048576).toFixed(1)} MiB adjacency`);}
  }
  const finalStat=fs.statSync(boundary.file);if(finalStat.size!==stat.size||finalStat.mtimeMs!==stat.mtimeMs)throw Error('Input changed during scan');
  const rows=[];for(const[direction,map]of[['out',outgoing],['in',incoming]])for(const[target,bucket]of map){const blocks=options.all?[...bucket].sort((a,b)=>a.start-b.start):bucket.sorted();for(const b of blocks)rows.push({...b,graph:group,direction,target});}
  const rowsFile=new BufferedFile(path.join(folder,complete?'selected_blocks.jsonl':'partial_blocks.jsonl'));try{for(const row of rows)rowsFile.write(Buffer.from(JSON.stringify(row)+'\n'));}finally{rowsFile.close();}
  const result={graph:group,complete,stop_reason:stop,source_file:boundary.file,source_file_bytes:stat.size,source_mtime_ms:stat.mtimeMs,adjacency_start:adjStart,scan_end:r.pos,
    edges_seen:edges,committed_source_blocks:roots,node_count_expected:boundary.sampled_nodes,
    node_and_adjacency_source_counts_match:complete?roots===boundary.sampled_nodes:null,
    node_minus_adjacency_source_count:complete?boundary.sampled_nodes-roots:null,
    structural_audit_required:complete&&roots!==boundary.sampled_nodes,
    elapsed_seconds:(Date.now()-startTime)/1000,adjacency_mib:(r.pos-adjStart)/1048576,
    target_source_addresses_found:[...sourceTargets].sort(),selected_block_count:rows.length,string_entries:strings.count,string_aliases:strings.aliasCount,
    strings_logged:options.log,all_target_incident_pair_blocks:options.all,counterparty_blocks_per_target_direction:options.all?null:options.k,selection_seed:options.seed,
    sampling:options.all?'all serialized pair blocks incident to each target; no pair-block sampling':'deterministic bottom-k of source/pair-block serialization offsets; technical counterparty-block sample, not transaction-uniform',
    model_training_performed:false};
  fs.writeFileSync(path.join(folder,'index_report.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));return result;
 }catch(e){fs.writeFileSync(path.join(folder,'ERROR.json'),JSON.stringify({graph:group,error:e.message,position:r.pos,edges_seen:edges,complete:false},null,2));throw e;}
 finally{r.close();strings.close();}
}

const args={};for(let i=2;i<process.argv.length;i++){const k=process.argv[i];if(k==='--log-strings'||k==='--all-target-blocks')args[k]=true;else args[k]=process.argv[++i];}
if(!args['--boundaries']||!args['--targets']||!args['--output-dir'])throw Error('Required: --boundaries --targets --output-dir');
const output=path.resolve(args['--output-dir']);if(fs.existsSync(output))throw Error('Output directory exists; use a new one');fs.mkdirSync(output,{recursive:true});
const boundaries=JSON.parse(fs.readFileSync(args['--boundaries'],'utf8')), plan=JSON.parse(fs.readFileSync(args['--targets'],'utf8'));
const k=Number(args['--blocks-per-direction']??16), max=Number(args['--max-adj-mib']??8)*1048576,seed=Number(args['--seed']??20261029);
if(!Number.isInteger(k)||k<1||k>64||!Number.isFinite(max)||max<0)throw Error('Invalid limits');
const groups=(args['--graphs']??Object.keys(boundaries).join(',')).split(',');const results=[];
for(const group of groups){if(!boundaries[group])throw Error('Unknown graph');console.log(`Indexing ${group} (${max?'bounded preflight':'complete outgoing adjacency'})...`);results.push(indexGraph(group,boundaries[group],plan.targets,path.join(output,group),{k,max,seed,log:Boolean(args['--log-strings']),all:Boolean(args['--all-target-blocks'])}));}
fs.writeFileSync(path.join(output,'summary.json'),JSON.stringify({graphs:results,code_sha256:crypto.createHash('sha256').update(fs.readFileSync(new URL(import.meta.url))).digest('hex'),target_plan_sha256:crypto.createHash('sha256').update(fs.readFileSync(args['--targets'])).digest('hex')},null,2));
