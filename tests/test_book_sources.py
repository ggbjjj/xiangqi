import json

import pytest
torch=pytest.importorskip('torch')

from board import Board
from data_pipeline.database import connect
from data_pipeline.loader import BOOK_SOURCE_LABELS,GameDataset,iter_game_batches
from data_pipeline.merge_cbl import identity
from data_pipeline.notation import STANDARD_FEN,to_fen
from encoder import move_to_action
from training.data import BookDataset,collate_examples


def insert(conn,moves,*,fen=STANDARD_FEN,result='1-0',split='train'):
    actions=[move_to_action(move) for move in moves]
    key,group,_=identity(fen,actions)
    conn.execute('INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?)',
        (key,fen,json.dumps(actions),len(actions),result,'book_study',split,group,'{}'))
    return key


@pytest.fixture
def libraries(tmp_path):
    main=tmp_path/'main.sqlite3'; endgames=tmp_path/'endgames.sqlite3'
    red0=(6,0,5,0); black0=(3,0,4,0); red2=(6,2,5,2); black2=(3,2,4,2)
    c=connect(main)
    a=insert(c,[red0,black0,red2])
    b=insert(c,[red2],result='*')
    v=insert(c,[(6,4,5,4),(3,4,4,4)],split='validation')
    insert(c,[(6,6,5,6),(3,6,4,6)],split='test')
    c.commit(); c.close()
    c=connect(endgames)
    assert insert(c,[red0,black0,red2],result='0-1')==a
    fen=to_fen(Board.fresh().apply_move(*red0),'b')
    e=insert(c,[black0,red2,black2,(6,4,5,4)],fen=fen,result='0-1')
    insert(c,[(3,4,4,4)],fen=to_fen(Board.fresh().apply_move(6,4,5,4),'b'),split='validation')
    insert(c,[(3,6,4,6)],fen=to_fen(Board.fresh().apply_move(6,6,5,6),'b'),split='test')
    c.commit(); c.close()
    return main,endgames,{'a':a,'b':b,'e':e,'v':v}


@pytest.mark.parametrize('mode,games,segments,steps',[('main',2,3,4),('endgames',2,4,7),('both',3,5,8)])
def test_all_modes_share_crop_padding_and_value_masks(libraries,mode,games,segments,steps):
    main,endgames,keys=libraries
    before=[path.read_bytes() for path in (main,endgames)]
    dataset=BookDataset(T=2,db=main,source=mode,endgame_db=endgames)
    assert len(dataset.ids)==games and len(dataset)==segments
    assert dataset.summary['mode']==mode and dataset.summary['games']==games
    samples=[dataset[i] for i in range(len(dataset))]
    assert sum(s.length for s in samples)==steps
    batch=collate_examples(samples)
    assert batch.loss.counts().book==steps
    for sample in samples:
        assert not sample.x[sample.length:].any()
        assert (sample.actions[sample.length:]==-100).all()
        assert (sample.value_targets[sample.length:]==-100).all()
        if mode=='both' and sample.source_game_id==keys['a']:
            assert sample.result=='*' and (sample.value_targets==-100).all()
    if mode=='both':
        assert dataset.summary['duplicate_games']==dataset.summary['value_conflicts']==1
        assert dataset.summary['by_source']=={'main':{'games':2,'segments':3},'endgames':{'games':1,'segments':2}}
    # The public batch iterator accepts the same source selection.
    batches=list(iter_game_batches(2,2,db=main,source=mode,endgame_db=endgames,shuffle=False))
    assert sum(int(lengths.sum()) for _,_,lengths in batches)==steps
    assert [path.read_bytes() for path in (main,endgames)]==before


def test_single_source_does_not_open_other_database_and_limit_follows_dedup(libraries,tmp_path):
    main,endgames,keys=libraries
    assert len(GameDataset(2,db=main,endgame_db=tmp_path/'absent.sqlite3'))==3
    assert len(GameDataset(2,db=tmp_path/'absent.sqlite3',endgame_db=endgames,source='endgames'))==4
    whole=GameDataset(2,db=main,endgame_db=endgames,source='both')
    limited=GameDataset(2,db=main,endgame_db=endgames,source='both',limit=1)
    assert limited.ids==whole.ids[:1]
    assert limited.segments==[item for item in whole.segments if item[0]==limited.ids[0]]
    with pytest.raises(ValueError,match='来源'):
        GameDataset(2,db=main,source='bad')
    with pytest.raises(ValueError,match='不同'):
        GameDataset(2,db=main,endgame_db=main,source='both')


def test_both_deduplicates_before_split_and_respects_main_split_groups(libraries):
    main,endgames,keys=libraries
    c=connect(main)
    c.execute("UPDATE games SET split='validation' WHERE game_id=?",(keys['a'],))
    group=c.execute('SELECT split_group FROM games WHERE game_id=?',(keys['v'],)).fetchone()[0]
    c.commit(); c.close()
    c=connect(endgames)
    c.execute('UPDATE games SET split_group=? WHERE game_id=?',(group,keys['e']))
    c.commit(); c.close()
    train=GameDataset(2,db=main,endgame_db=endgames,source='both')
    validation=GameDataset(2,'validation',main,endgame_db=endgames,source='both')
    assert keys['a'] not in train.ids and keys['e'] not in train.ids
    assert keys['a'] in validation.ids and keys['e'] in validation.ids
    assert not set(train.ids)&set(validation.ids)
    assert validation.summary['duplicate_games']==1


def test_both_refuses_same_id_with_different_content(libraries):
    main,endgames,keys=libraries
    c=connect(endgames)
    c.execute('DELETE FROM games WHERE game_id=?',(keys['a'],))
    c.execute('UPDATE games SET game_id=? WHERE game_id=?',(keys['a'],keys['e']))
    c.commit(); c.close()
    with pytest.raises(ValueError,match='走法不一致'):
        GameDataset(2,db=main,endgame_db=endgames,source='both')


@pytest.mark.parametrize('mode,games,segments',[('main',2,3),('endgames',2,4),('both',3,5)])
def test_training_cli_loads_mode_for_train_validation_and_records_it(libraries,tmp_path,capsys,mode,games,segments):
    from model_store import save_checkpoint,load_checkpoint
    from models.Opus.config import OpusConfig
    from models.Opus.network import Opus5
    from training.__main__ import main
    torch.set_num_threads(2)
    model=Opus5(OpusConfig(dim=16,heads=(4,2,1,2,1),context_windows=(2,3,4,3,4),
                          ffn_dim=32,cnn_channels=8,value_hidden=8))
    db=tmp_path/'models.sqlite3'
    save_checkpoint(model,path=db,run_id='initial',epoch=0,global_step=0,metrics={})
    books,endgames,_=libraries
    report=main(['--family','Opus','--source','books','--book-source',mode,
        '--book-db',str(books),'--endgame-db',str(endgames),'--model-db',str(db),
        '--T','2','--epochs','1','--batch-size','2','--micro-batch','1','--device','cpu',
        '--output',str(tmp_path/'runs'),'--no-progress','--smoke'])
    config=json.loads((tmp_path/'runs'/report['run_id']/'config.json').read_text(encoding='utf-8'))
    assert config['book_games']==games and config['book_segments']==segments
    assert config['book_source']==config['book_selection']['mode']==config['validation_selection']['mode']==mode
    assert report['best']['validation']['book_selection']['mode']==mode
    state=load_checkpoint(db,report['checkpoints'][0],resume=True)[2]
    assert state['book_source']==mode and state['book_selection']['segments']==segments
    output=capsys.readouterr().out
    assert f'棋谱读取方式：{BOOK_SOURCE_LABELS[mode]}（--book-source {mode}）' in output
    assert f'本次训练总局数（切段后）：{segments}' in output


def test_legacy_training_entry_uses_the_same_both_reader(libraries,tmp_path,capsys):
    from model_store import save_checkpoint
    from models.HaiKu.config import Haiku5Config
    from models.HaiKu.network import Haiku5
    import train
    torch.set_num_threads(2)
    model=Haiku5(Haiku5Config(dim=16,heads=4,ffn_dim=32,cnn_channels=8,context_plies=2))
    db=tmp_path/'models.sqlite3'
    save_checkpoint(model,path=db,run_id='initial',epoch=0,global_step=0,metrics={})
    main,endgames,_=libraries
    train.main(['--data',str(main),'--endgame-db',str(endgames),'--book-source','both',
        '--model-db',str(db),'--T','2','--epochs','1','--batch-size','2','--micro-batch','1',
        '--device','cpu','--precision','none','--output',str(tmp_path/'runs'),'--run-name','legacy',
        '--no-progress','--smoke'])
    config=json.loads((tmp_path/'runs/legacy/config.json').read_text(encoding='utf-8'))
    assert config['book_source']==config['book_selection']['mode']=='both'
    assert config['train_games']==3 and config['train_segments']==5
    assert '棋谱读取方式：主棋谱库＋残局库（--book-source both）' in capsys.readouterr().out
