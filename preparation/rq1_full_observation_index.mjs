// Index every available outgoing source block and incoming pair block for an
// explicit account cohort. No sampling, unpickling, feature fitting or training.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {Reader, opcode} from './rq1_pickle_primitives.mjs';
const NUM=Symbol('number'),OBJ=Symbol('opaque'),MARK=Symbol('mark');
const codeHash=crypto.createHash('sha256').update(fs.readFileSync(new URL(import.meta.url))).digest('hex');
const json=file=>JSON.parse(fs.readFileSync(file,'utf8').replace(/^\uFEFF/,''));
class Bits {
 constructor(){this.b=new Uint8Array(8*1024*1024);}
 has(i){return i<this.b.length*8&&Boolean(this.b[i>>>3]&(1<<(i&7)));}
 set(i){if(!Number.isInteger(i)||i<0||i>=2**32)throw Error('Memo overflow');while(i>=this.b.length*8){const b=new Uint8Array(this.b.length*2);b.set(this.b);this.b=b;}this.b[i>>>3]|=1<<(i&7);}
}
class Writer {
 constructor(file){this.fd=fs.openSync(file,'wx');this.b=Buffer.allocUnsafe(1024*1024);this.n=0;this.offset=0;}
 write(value){let i=0;while(i<value.length){const take=Math.min(this.b.length-this.n,value.length-i);value.copy(this.b,this.n,i,i+take);this.n+=take;this.offset+=take;i+=take;if(this.n===this.b.length)this.flush();}}
 row(value){this.write(Buffer.from(JSON.stringify(value)+'\n'));}
 flush(){let i=0;while(i<this.n)i+=fs.writeSync(this.fd,this.b,i,this.n-i);this.n=0;}
 close(){this.flush();fs.closeSync(this.fd);}
}
class Frame {
 constructor(start,memo){this.start=start;this.end=start+1;this.memo=memo;this.tag='empty';this.edges=0;this.pairs=0;this.hasIncoming=false;}
}
function parseCSV(text){
 const rows=[];let row=[],field='',quoted=false;
 for(let i=0;i<text.length;i++){const c=text[i];if(quoted){if(c==='"'){if(text[i+1]==='"'){field+='"';i++;}else quoted=false;}else field+=c;}
 else if(c==='"'){if(field)throw Error('Invalid CSV quoting');quoted=true;}
 else if(c===','){row.push(field);field='';}
 else if(c==='\r'||c==='\n'){if(c==='\r'&&text[i+1]==='\n')i++;row.push(field);if(row.some(v=>v!==''))rows.push(row);row=[];field='';}
 else field+=c;}
 if(quoted)throw Error('Unterminated CSV field');if(field||row.length){row.push(field);rows.push(row);}return rows;
}
function loadCohort(file,labelsFile,expected){
 const rows=parseCSV(fs.readFileSync(file,'utf8').replace(/^\uFEFF/,'')),header=rows.shift();
 const positions=Object.fromEntries(header.map((key,i)=>[key,i]));
 for(const key of ['address','class_id','class_name','label_status'])if(!(key in positions))throw Error('Missing CSV column '+key);
 const labels=new Map(json(labelsFile).accounts.map(r=>[r.address,r]));const map=new Map(),counts={normal:0,phish:0,scams:0},ids={normal:0,phish:1,scams:2};
 for(const row of rows){if(row.length!==header.length)throw Error('CSV width differs');const a=row[positions.address],name=row[positions.class_name],label=labels.get(a);
  if(!/^0x[0-9a-f]{40}$/.test(a)||map.has(a))throw Error('Invalid or duplicated cohort address');
  if(!label||label.status!=='candidate_without_detected_conflict'||label.source_declared_class!==name||row[positions.label_status]!==label.status||String(ids[name])!==row[positions.class_id])throw Error('Cohort/explicit-label mismatch');
  map.set(a,name);counts[name]++;
 }
 if(map.size!==expected)throw Error(`Expected ${expected} accounts, got ${map.size}`);
 if(expected===4913&&(counts.normal!==636||counts.phish!==1165||counts.scams!==3112))throw Error('Class counts differ from prepared cohort');
 return {map,counts};
}
function indexGraph(group,boundary,cohort,folder,maxBytes){
 fs.mkdirSync(folder);const before=fs.statSync(boundary.file);if(before.size!==boundary.file_bytes)throw Error('Raw file size differs from audited boundary');
 const outputs={};for(const name of ['incoming_pairs.jsonl','outgoing_sources.jsonl','source_context.jsonl','strings.idx','strings.dat','strings.aliases'])outputs[name]=new Writer(path.join(folder,name));
 const r=new Reader(boundary.file),dicts=new Bits(),selected=new Map(),coverage=new Map([...cohort].map(([a,name])=>[a,{address:a,class_name:name,raw_source_entries:0,outgoing_pair_blocks:0,outgoing_edge_records:0,incoming_pair_blocks:0,incoming_edge_records:0}]));
 let last=OBJ,nextMemo=0,nodesDone=false,root=null,stack=[],complete=false,roots=0,edges=0,inPairs=0,inEdges=0,outSources=0,stringEntries=0,lastProgress=0;
 const started=Date.now();
 function selectedAddress(key){const s=typeof key==='string'?key:key?.kind==='memo'?selected.get(key.index):null;return s&&cohort.has(s.toLowerCase())?s.toLowerCase():null;}
 function keyRef(key){if(typeof key==='string')return {kind:'literal',value:key};if(key?.kind==='memo')return key;throw Error('Non-string endpoint key');}
 function saveMemo(i,value){
  if(value instanceof Frame||value===OBJ){dicts.set(i);if(value instanceof Frame)value.end=r.pos;}
  else if(typeof value==='string'){
   if(cohort.has(value.toLowerCase()))selected.set(i,value);
   const raw=Buffer.from(value),b=Buffer.allocUnsafe(16);b.writeUInt32LE(i,0);b.writeUInt32LE(raw.length,4);b.writeBigUInt64LE(BigInt(outputs['strings.dat'].offset),8);outputs['strings.idx'].write(b);outputs['strings.dat'].write(raw);stringEntries++;
  }else if(value?.kind==='memo'){const b=Buffer.allocUnsafe(8);b.writeUInt32LE(i,0);b.writeUInt32LE(value.index,4);outputs['strings.aliases'].write(b);if(selected.has(value.index))selected.set(i,selected.get(value.index));}
 }
 function assign(parent,key,value){
  if(!(parent instanceof Frame))throw Error('Invalid assignment target');parent.end=r.pos;
  if(parent===root){
   if(!(value instanceof Frame)||!['neighbors','empty'].includes(value.tag))throw Error('Invalid source dictionary');roots++;
   const target=selectedAddress(key);
   if(target){outSources++;const c=coverage.get(target);c.raw_source_entries++;c.outgoing_pair_blocks+=value.pairs;c.outgoing_edge_records+=value.edges;
    outputs['outgoing_sources.jsonl'].row({graph:group,address:target,source:keyRef(key),start:value.start,end:value.end,memo_start:value.memo,pair_blocks:value.pairs,edge_records:value.edges});}
   if(value.hasIncoming||target)outputs['source_context.jsonl'].row({source_block_start:value.start,source:keyRef(key)});
   return;
  }
  if(value instanceof Frame){
   if(value.tag==='leaf'||(value.tag==='empty'&&key===NUM)){
    if(!['empty','pair'].includes(parent.tag))throw Error('Mixed edge dictionary shape');parent.tag='pair';parent.edges++;edges++;
   }else if(['pair','empty'].includes(value.tag)){
    if(!['empty','neighbors'].includes(parent.tag))throw Error('Invalid adjacency depth');parent.tag='neighbors';parent.pairs++;parent.edges+=value.edges;
    const target=selectedAddress(key);
    if(target){parent.hasIncoming=true;inPairs++;inEdges+=value.edges;const c=coverage.get(target);c.incoming_pair_blocks++;c.incoming_edge_records+=value.edges;
     outputs['incoming_pairs.jsonl'].row({graph:group,address:target,source_block_start:parent.start,target:keyRef(key),start:value.start,end:value.end,memo_start:value.memo,edge_records:value.edges});}
   }else throw Error('Unsupported nested attributes');
  }else{if(!['empty','leaf'].includes(parent.tag))throw Error('Nonflat attributes');parent.tag='leaf';}
 }
 try{
  for(;;){
   if(!nodesDone&&r.pos===boundary.last_opcode_position){if(nextMemo!==boundary.next_memo_index_at_boundary)throw Error('Node memo boundary differs');nodesDone=true;console.log(`${group}: node prefix verified`);}
   if(!nodesDone&&r.pos>boundary.last_opcode_position)throw Error('Missed node boundary');
   if(nodesDone&&maxBytes>0&&r.pos-boundary.last_opcode_position>=maxBytes)break;
   const o=opcode(r);if(o.op==='HEADER')continue;
   if(o.op==='MEMO'){const i=o.index??nextMemo;saveMemo(i,nodesDone&&root?stack.at(-1):last);nextMemo=Math.max(nextMemo,i+1);continue;}
   if(!nodesDone){last=o.op==='STRING'?o.v:o.op==='GET'?(dicts.has(o.index)?OBJ:{kind:'memo',index:o.index}):o.op==='NUMBER'?NUM:OBJ;continue;}
   if(!root){if(o.op==='STRING'&&o.v==='_adj'){last=o.v;continue;}if(o.op!=='DICT')throw Error('Invalid adjacency entry');root=new Frame(o.pos,nextMemo);stack=[root];continue;}
   if(o.op==='STRING'){if(stack.length===1&&['_succ','_pred','adj'].includes(o.v)){complete=true;break;}stack.push(o.v);}
   else if(o.op==='NUMBER')stack.push(NUM);
   else if(o.op==='GET'){if(dicts.has(o.index))throw Error('Shared dictionary reference needs explicit support');stack.push({kind:'memo',index:o.index});}
   else if(o.op==='DICT')stack.push(new Frame(o.pos,nextMemo));
   else if(o.op==='MARK')stack.push(MARK);
   else if(o.op==='SET'){if(stack.length===1){complete=true;break;}const v=stack.pop(),k=stack.pop();assign(stack.at(-1),k,v);}
   else if(o.op==='SETS'){let j=stack.length-1;while(j>=0&&stack[j]!==MARK)j--;if(j<1||(stack.length-j-1)%2)throw Error('Malformed batch');for(let i=j+1;i<stack.length;i+=2)assign(stack[j-1],stack[i],stack[i+1]);stack.length=j;}
   else throw Error(`Unsupported ${o.op} inside adjacency`);
   if(edges-lastProgress>=5000000){lastProgress=edges;console.log(`${group}: ${(edges/1000000).toFixed(1)} million source edge records scanned`);}
  }
  const after=fs.statSync(boundary.file);if(before.size!==after.size||before.mtimeMs!==after.mtimeMs)throw Error('Raw input changed');
  const report={graph:group,complete,index_only:true,sampling_performed:false,source_file:boundary.file,source_file_bytes:before.size,source_mtime_ms:before.mtimeMs,
   node_entries:boundary.sampled_nodes,source_entries:roots,source_edge_records_scanned:edges,outgoing_target_source_blocks:outSources,incoming_target_pair_blocks:inPairs,incoming_target_edge_record_occurrences:inEdges,
   scan_end:r.pos,string_entries:stringEntries,elapsed_seconds:(Date.now()-started)/1000,
   note:'Incoming and outgoing selections may overlap. Counts are serialized graph observations, not unique blockchain transactions. Features and training data have not been materialized.',
   full_features_extracted:false,model_training_performed:false};
  fs.writeFileSync(path.join(folder,'coverage.json'),JSON.stringify([...coverage.values()],null,2));
  fs.writeFileSync(path.join(folder,'index_report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));return report;
 }catch(error){fs.writeFileSync(path.join(folder,'ERROR.json'),JSON.stringify({complete:false,error:error.message,position:r.pos},null,2));throw error;}
 finally{r.close();for(const writer of Object.values(outputs))writer.close();}
}
function main(){
 const args={};for(let i=2;i<process.argv.length;i+=2)args[process.argv[i]]=process.argv[i+1];
 for(const key of ['--accounts','--labels','--boundaries','--output-dir'])if(!args[key])throw Error('Missing '+key);
 const expected=Number(args['--expected-accounts']??4913),maxMiB=Number(args['--max-adj-mib']??0);
 if(!Number.isInteger(expected)||expected<1||!Number.isFinite(maxMiB)||maxMiB<0)throw Error('Invalid limits');
 const cohort=loadCohort(args['--accounts'],args['--labels'],expected),boundaries=json(args['--boundaries']);
 const folder=path.resolve(args['--output-dir']);if(fs.existsSync(folder))throw Error('Output exists; choose a new directory');fs.mkdirSync(folder,{recursive:true});
 const graphs=(args['--graphs']??Object.keys(boundaries).join(',')).split(','),reports=[];
 for(const graph of graphs){if(!boundaries[graph]?.node_table_complete)throw Error('Missing complete boundary '+graph);reports.push(indexGraph(graph,boundaries[graph],cohort.map,path.join(folder,graph),maxMiB*1048576));}
 const inputHashes={};for(const name of ['--accounts','--labels','--boundaries'])inputHashes[name]=crypto.createHash('sha256').update(fs.readFileSync(args[name])).digest('hex');
 fs.writeFileSync(path.join(folder,'summary.json'),JSON.stringify({status:reports.every(r=>r.complete)?'complete_observation_ranges_indexed':'preflight_only',
  cohort_accounts:cohort.map.size,class_counts:cohort.counts,graphs:reports,input_sha256:inputHashes,code_sha256:codeHash,
  primitive_reader_sha256:crypto.createHash('sha256').update(fs.readFileSync(new URL('./rq1_pickle_primitives.mjs',import.meta.url))).digest('hex'),full_features_extracted:false,model_training_performed:false},null,2));
}
main();
