"""Serve live training curves with Chinese axes; TensorBoard remains optional."""
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import parse_qs,urlparse
import webbrowser

from project_paths import ROOT
from .control import StopController
from .dashboard_data import CurveReader


def make_server(logdir,*,host='127.0.0.1',port=6006):
    reader=CurveReader(logdir)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed=urlparse(self.path)
            try:
                if parsed.path in ('/','/index.html'):
                    self.send_bytes(Path(__file__).with_name('dashboard.html').read_bytes(),'text/html; charset=utf-8')
                elif parsed.path=='/api/runs': self.send_json({'runs':reader.runs()})
                elif parsed.path=='/api/scalars':
                    self.send_json(reader.read(parse_qs(parsed.query).get('run',[''])[0]))
                else: self.send_json({'error':'页面不存在'},404)
            except ValueError as error: self.send_json({'error':str(error)},400)
            except (OSError,RuntimeError) as error: self.send_json({'error':f'读取曲线失败：{error}'},500)

        def send_json(self,data,status=200):
            self.send_bytes(json.dumps(data,ensure_ascii=False,allow_nan=False).encode('utf-8'),
                            'application/json; charset=utf-8',status)

        def send_bytes(self,data,content_type,status=200):
            self.send_response(status); self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(data))); self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff'); self.end_headers()
            try: self.wfile.write(data)
            except (BrokenPipeError,ConnectionResetError): pass

        def log_message(self,*args): pass
    return ThreadingHTTPServer((host,port),Handler)


def launch_tensorboard(logdir,*,host='127.0.0.1',port=6006):
    from tensorboard import program
    board=program.TensorBoard()
    board.configure(argv=['tensorboard','--logdir',str(Path(logdir).resolve()),'--host',host,
        '--port',str(port),'--reload_interval','1','--load_fast','false'])
    return board.launch()


def main(argv=None):
    p=argparse.ArgumentParser(description='实时训练曲线窗口，中文坐标轴；训练停止后仍可查看历史曲线')
    p.add_argument('--logdir',type=Path,default=ROOT/'runs',help='可选所有运行目录，或只选一次训练的输出目录')
    p.add_argument('--host',default='127.0.0.1',help='本地默认回环地址；AutoDL 端口映射可指定 0.0.0.0')
    p.add_argument('--port',type=int,default=6006)
    p.add_argument('--open',action='store_true',help='本地自动打开默认浏览器；AutoDL 无桌面时省略')
    p.add_argument('--tensorboard',action='store_true',help='可选：改用原版 TensorBoard，其内置坐标控件为英文')
    args=p.parse_args(argv)
    if not 1<=args.port<=65535: p.error('port 必须在 1..65535 内')
    server=None
    try:
        with StopController() as stop:
            if args.tensorboard: url=launch_tensorboard(args.logdir,host=args.host,port=args.port)+'#scalars'
            else:
                server=make_server(args.logdir,host=args.host,port=args.port)
                threading.Thread(target=server.serve_forever,daemon=True,name='xiangqi-curves').start()
                host='127.0.0.1' if args.host=='0.0.0.0' else args.host
                if ':' in host: host=f'[{host}]'
                url=f'http://{host}:{server.server_port}/'
            print(f'实时曲线窗口：{url}\n日志目录：{args.logdir.resolve()}\n每秒刷新；Ctrl+C 仅关闭曲线服务，不中断训练。',flush=True)
            if args.open: webbrowser.open(url)
            while not stop.is_set(): stop.event.wait(.5)
    except (OSError,ValueError) as error: p.exit(1,f'{error}\n')
    finally:
        if server is not None: server.shutdown(); server.server_close()


if __name__=='__main__': main()
