"""Gemini streaming transport for the same configured model and image parts.

One POST, no retry or redirect. Preserve received chunks before aggregation;
never fabricate STOP or image bytes. A midstream disconnect remains unknown.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
import traceback
from urllib.request import Request,build_opener,ProxyHandler

from .core import write,require


def aggregate(chunks):
    require(chunks and all(isinstance(c,dict) for c in chunks),"Empty/invalid Gemini stream")
    parts=[];finish=None;last={}
    for chunk in chunks:
        if chunk.get("promptFeedback",{}).get("blockReason"):
            return chunk
        for candidate in chunk.get("candidates",[]):
            require(candidate.get("index",0)==0,"Multiple Gemini stream candidates not supported")
            parts.extend(candidate.get("content",{}).get("parts",[]))
            if candidate.get("finishReason") is not None:finish=candidate["finishReason"]
        for key in ["usageMetadata","modelVersion","responseId","promptFeedback"]:
            if key in chunk:last[key]=chunk[key]
    require(finish is not None,"No final provider finishReason; incomplete stream")
    return {**last,"candidates":[{"index":0,"content":{"role":"model","parts":parts},"finishReason":finish}]}


def transport(stage):
    from data_generation_methods.r2h_pipeline.providers import _NoRedirect,ProviderError
    stage=Path(stage)
    def call(method,url,*,headers,json:dict|None,timeout):
        require(method=="POST" and url.endswith(":streamGenerateContent?alt=sse"),"Unexpected stream request")
        body=__import__('json').dumps(json,ensure_ascii=False,allow_nan=False,separators=(",",":")).encode()
        require(len(body)<=20*1024*1024,"Stream request too large")
        public_headers={**headers,"Accept":"text/event-stream","User-Agent":"Mozilla/5.0","Connection":"close"}
        req=Request(url,data=body,headers=public_headers,method="POST")
        started=time.time();chunks=[];events=[];total=0
        try:
            with build_opener(_NoRedirect(),ProxyHandler({})).open(req,timeout=timeout) as response:
                write(stage/"response_headers_public.json",{"http_status":response.status,"content_type":response.headers.get("Content-Type"),"first_headers_seconds":time.time()-started})
                content_type=response.headers.get("Content-Type","")
                if "text/event-stream" not in content_type:
                    raw=response.read(128*1024*1024+1)
                    require(len(raw)<=128*1024*1024,"Response too large")
                    parsed=__import__('json').loads(raw)
                    chunks=parsed if isinstance(parsed,list) else [parsed]
                    write(stage/"non_sse_provider_response.json",parsed)
                else:
                    data_lines=[]
                    for raw in response:
                        total+=len(raw);require(total<=128*1024*1024,"Response too large")
                        line=raw.decode("utf-8").rstrip("\r\n")
                        if line.startswith("data:"):data_lines.append(line[5:].lstrip())
                        elif line=="" and data_lines:
                            data="\n".join(data_lines);data_lines=[]
                            if data=="[DONE]":break
                            value=__import__('json').loads(data)
                            chunks.extend(value if isinstance(value,list) else [value])
                            # Durable chunks contain only provider response data, no credentials/headers.
                            write(stage/"stream_chunks"/f"chunk_{len(events):04d}.json",value)
                            events.append({"chunk":len(events),"elapsed_seconds":time.time()-started})
                    if data_lines:
                        data="\n".join(data_lines)
                        if data!="[DONE]":
                            value=__import__('json').loads(data);chunks.extend(value if isinstance(value,list) else [value])
            result=aggregate(chunks)
            write(stage/"provider_response.json",result)
            write(stage/"stream_timing.json",{"single_post":True,"chunks":len(chunks),"events":events,"elapsed_seconds":time.time()-started,"automatic_retry":False})
            return result
        except Exception as exc:
            write(stage/"stream_failure_detail.json",{"exception_type":type(exc).__name__,"reason_type":type(getattr(exc,"reason",None)).__name__,
                "http_status":getattr(exc,"code",None),"traceback_function_names":[f.name for f in traceback.extract_tb(exc.__traceback__)],
                "received_chunks":len(chunks),"elapsed_seconds":time.time()-started,"single_post":True,"automatic_retry":False})
            raise
    return call


def client(config,stage):
    from data_generation_methods.r2h_pipeline.providers import GeminiClient,_identifier,_json_bytes,ProviderError
    class StreamingGemini(GeminiClient):
        def _generate(self,model,parts,generation_config):
            _identifier(model)
            body={"contents":[{"role":"user","parts":parts}],"generationConfig":generation_config}
            if len(_json_bytes(body))>20*1024*1024:raise ProviderError("inline_limit_exceeded")
            return self._call("POST",self.base_url+"/models/"+model+":streamGenerateContent?alt=sse",body)
    return StreamingGemini(config,transport=transport(stage))
