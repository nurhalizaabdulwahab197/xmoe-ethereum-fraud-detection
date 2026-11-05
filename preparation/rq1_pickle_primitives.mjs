import fs from 'node:fs';
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
  case 0x4a:return {op:'NUMBER',v:r.bytes(4).readInt32LE(0),pos};case 0x4b:return {op:'NUMBER',v:r.byte(),pos};case 0x4d:return {op:'NUMBER',v:r.bytes(2).readUInt16LE(0),pos};case 0x47:return {op:'NUMBER',v:r.bytes(8).readDoubleBE(0),pos};
  case 0x8a:r.skip(r.byte());return {op:'NUMBER',pos};case 0x8b:r.skip(r.u32());return {op:'NUMBER',pos};
  case 0x49:case 0x4c:case 0x46:r.line();return {op:'NUMBER',pos};case 0x4e:case 0x88:case 0x89:return {op:'NUMBER',pos};
  case 0x63:r.line();r.line();return {op:'OPAQUE',pos};
  case 0x82:r.skip(1);return {op:'OPAQUE',pos};case 0x83:r.skip(2);return {op:'OPAQUE',pos};case 0x84:r.skip(4);return {op:'OPAQUE',pos};
  case 0x81:case 0x85:case 0x86:case 0x87:case 0x93:case 0x92:case 0x62:case 0x52:case 0x29:case 0x74:case 0x5d:case 0x6c:case 0x61:case 0x65:case 0x30:case 0x31:case 0x32:case 0x8f:case 0x90:case 0x91:return {op:'OPAQUE',pos};
  case 0x42:r.skip(r.u32());return {op:'OPAQUE',pos};case 0x43:r.skip(r.byte());return {op:'OPAQUE',pos};case 0x8e:case 0x96:r.skip(r.u64());return {op:'OPAQUE',pos};
  default:throw Error(`Unsupported opcode 0x${c.toString(16)} at ${pos}`);
}}

export {Reader, opcode};

