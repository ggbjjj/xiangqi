"""Resume the checkpoint recorded in TrainingRecords/current_resume.json."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from project_paths import ROOT


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT/path


def validate_record(config):
    """Read-only verification; also verify that optimizer state can be restored."""
    context_plies = config.get('context_plies')
    if context_plies is not None and (type(context_plies) is not int or context_plies <= 0):
        raise ValueError('context_plies 必须为正整数')
    import torch
    from data_pipeline.loader import GameDataset
    from model_store import load_checkpoint
    model,meta,state = load_checkpoint(project_path(config['model_db']),config['checkpoint_id'],
                                       'cpu',resume=True)
    if (meta['epoch'] != config['completed_epoch'] or meta['run_id'] != config['run_id']
            or meta['sha256'] != config['weights_sha256']):
        raise ValueError('检查点与保存的续训记录不一致')
    if state.get('T') != config['T'] or state.get('seed') != config['seed']:
        raise ValueError('续训 T 或 seed 与检查点不一致')
    datasets = [GameDataset(config['T'],split,project_path(config['data']),config['limit_games'])
                for split in ('train','validation','test')]
    if not all(len(dataset) for dataset in datasets):
        raise ValueError('训练、验证或测试划分为空')
    ids = [ident for dataset in datasets for ident in dataset.ids]
    fingerprint = hashlib.sha256('\n'.join(ids).encode()).hexdigest()
    if fingerprint != config['dataset_fingerprint'] or fingerprint != state.get('dataset_fingerprint'):
        raise ValueError('数据划分与检查点不一致，不能继续这次训练')
    if not state.get('optimizer',{}).get('state') or not isinstance(state.get('rng_state'),torch.Tensor):
        raise ValueError('检查点缺少优化器或随机数状态')
    optimizer = torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    optimizer.load_state_dict(state['optimizer'])
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError('检查点包含非有限模型参数')
    history_path = project_path(config['source_history'])
    if history_path.exists():
        with history_path.open(encoding='utf-8',newline='') as stream:
            rows = [row for row in csv.DictReader(stream) if int(row['epoch']) <= meta['epoch']]
        if not rows or rows[-1]['checkpoint_id'] != config['checkpoint_id']:
            raise ValueError('历史 CSV 与续训检查点不一致')
    else:
        # Parameters contain epoch metrics even if the remote upload omitted old logs.
        metrics = json.loads(meta['metrics_json'])
        validation = metrics['validation']
        rows = [{'epoch':meta['epoch'],'train_loss':metrics['train_loss'],
                 'validation_loss':validation['loss'],'validation_top1':validation['top1'],
                 'validation_top3':validation['top3'],'checkpoint_id':config['checkpoint_id']}]
    return meta,rows


def training_arguments(config, epochs, output, device, precision):
    options = {'resume':config['checkpoint_id'],'epochs':epochs,'model-db':project_path(config['model_db']),
               'data':project_path(config['data']),'output':output,'device':device,'precision':precision}
    for key in ('T','seed','batch_size','micro_batch','workers','cpu_threads','lr',
                'weight_decay','clip_grad','limit_games'):
        options[key.replace('_','-')] = config[key]
    if config.get('context_plies') is not None:
        options['context-plies'] = config['context_plies']
    return [part for key,value in options.items() for part in ('--'+key,str(value))]


def main(argv=None):
    parser = argparse.ArgumentParser(description='从项目当前续训记录的检查点继续训练；默认再完成1轮')
    parser.add_argument('--record',type=Path,default=ROOT/'TrainingRecords/current_resume.json')
    parser.add_argument('--epochs',type=int,help='目标总轮数，必须大于记录中已完成的轮数')
    parser.add_argument('--device',help='默认 cuda')
    parser.add_argument('--precision',choices=('auto','bf16','fp16','none'))
    parser.add_argument('--context-plies',type=int,
                        help='独立设置 KV 历史窗口；默认使用续训记录中的 context_plies，未记录时沿用检查点')
    parser.add_argument('--check-only',action='store_true',help='仅用 CPU 核验参数、优化器和数据，不开始训练')
    args = parser.parse_args(argv)
    if args.context_plies is not None and args.context_plies <= 0:
        parser.error('--context-plies 必须为正数')
    try:
        config = json.loads(args.record.read_text(encoding='utf-8'))
        if args.context_plies is not None:
            config['context_plies'] = args.context_plies
        epochs = config['default_total_epochs'] if args.epochs is None else args.epochs
        if epochs <= config['completed_epoch']:
            raise ValueError('目标总轮数必须大于已完成轮数')
        meta,rows = validate_record(config)
        context_plies = config.get('context_plies')
        if context_plies is None:
            context_plies = json.loads(meta['config_json'])['context_plies']
        device,precision = args.device or config['device'],args.precision or config['precision']
        output = project_path(config['output_base'])/datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
        arguments = training_arguments(config,epochs,output,device,precision)
        print(f'已核验：{config["public_model_name"]} / epoch={meta["epoch"]} / '
              f'optimizer step={meta["global_step"]} / checkpoint={config["checkpoint_id"]}',flush=True)
        print(f'继续第 {meta["epoch"]+1} 至 {epochs} 轮；T={config["T"]}；'
              f'KV 历史窗口={context_plies}；微批量 {config["micro_batch"]}；数据划分一致。',flush=True)
        if args.check_only:
            print('仅检查完成，未启动训练、未改写参数或日志。',flush=True)
            return arguments
        import torch
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise ValueError('未检测到 CUDA GPU；请在服务器 GPU 环境运行，或只使用 --check-only')
        run_output = output/config['run_id']
        run_output.mkdir(parents=True,exist_ok=False)
        with (run_output/'history.csv').open('w',encoding='utf-8',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        origin = {'source_record':str(args.record),'checkpoint_id':config['checkpoint_id'],
                  'completed_epoch_at_resume':meta['epoch'],'target_total_epochs':epochs,
                  'source_history':config['source_history'],'context_plies':context_plies}
        (run_output/'resume_origin.json').write_text(json.dumps(origin,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(f'新日志目录：{run_output}',flush=True)
    except (OSError,ValueError,KeyError) as error:
        parser.exit(1,f'续训检查失败：{error}\n')
    from train import main as train_main
    return train_main(arguments)


if __name__ == '__main__': main()
