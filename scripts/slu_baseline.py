# coding: utf-8

import sys, os, time, gc, json
from torch.optim import Adam, lr_scheduler

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)
sys.path.append('/mnt/workspace/Project/utils')
sys.path.append('/mnt/workspace/Project/model')
from args import init_args
from initialization import *
from example import Example
from batch import from_example_list
from vocab import PAD
from slu_baseline_tagging import SLUTagging, SLU_DE_Tagging
import logging

# initialization params, output path, logger, random seed and torch.device
args = init_args(sys.argv[1:])
set_random_seed(args.seed)
device = set_torch_device(args.device)
print("Initialization finished ...")
print("Random seed is set to %d" % (args.seed))
print("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

start_time = time.time()
train_path = os.path.join(args.dataroot, 'train.json')
dev_path = os.path.join(args.dataroot, 'development.json')
Example.configuration(args.dataroot, train_path=train_path, word2vec_path=args.word2vec_path)
train_dataset = Example.load_dataset(train_path)
dev_dataset = Example.load_dataset(dev_path)
if args.train_manual:
    train_manual_dataset = Example.load_dataset(train_path, use_manual=True)
print("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
print("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))

args.vocab_size = Example.word_vocab.vocab_size # 词表大小
#print('args.vocab_size: ',args.vocab_size) # NOTE
args.pad_idx = Example.word_vocab[PAD]
#print('args.pad_idx: ', args.pad_idx)
args.num_tags = Example.label_vocab.num_tags    # tag总数 74
#print('args.num_tags: ', args.num_tags)
args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)
#print('args.tag_pad_idx: ', args.tag_pad_idx)

if not args.pretrained_model:
    model = SLUTagging(args).to(device)
    Example.word2vec.load_embeddings(model.word_embed, Example.word_vocab, device=device) # 静态embedding
else:
    model = SLU_DE_Tagging(args).to(device)

if args.testing:
    model_path = os.path.join('/mnt/workspace/Project/weight', args.name, 'checkpoint', 'model.bin')
    check_point = torch.load(open('model.bin', 'rb'), map_location=device)
    model.load_state_dict(check_point['model'])
#model.load_state_dict(torch.load('/mnt/workspace/Project/weight/bert/checkpoint/model.bin'))
    print("Load saved model from root path")


def set_optimizer(model, args):
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grouped_params = [{'params': list(set([p for n, p in params]))}]
    optimizer = Adam(grouped_params, lr=args.lr)
    return optimizer


def decode(choice, manual = False):
    assert choice in ['train', 'dev']
    model.eval()
    if manual:
        dataset = train_manual_dataset if choice == 'train' else dev_dataset
    else:
        dataset = train_dataset if choice == 'train' else dev_dataset
    predictions, labels = [], []
    total_loss, count = 0, 0
    with torch.no_grad():
        for i in range(0, len(dataset), args.batch_size):
            cur_dataset = dataset[i: i + args.batch_size]
            current_batch = from_example_list(args, cur_dataset, device, train=True)
            pred, label, loss = model.decode(Example.label_vocab, current_batch)
            for j in range(len(current_batch)):
                if any([l.split('-')[-1] not in current_batch.utt[j] for l in pred[j]]):
                    print(current_batch.utt[j], pred[j], label[j])
            predictions.extend(pred)
            labels.extend(label)
            total_loss += loss
            count += 1
        metrics = Example.evaluator.acc(predictions, labels) 
        # 输入evaluator的predictions的格式和train.json中semantic的格式一致，不再是BIO标签了
        # metrics：用于评估模型性能的指标
    torch.cuda.empty_cache()
    gc.collect()
    return metrics, total_loss / count


def predict():
    model.eval()
    test_path = os.path.join(args.dataroot, 'test_unlabelled.json')
    test_dataset = Example.load_dataset(test_path)
    predictions = {}
    with torch.no_grad():
        for i in range(0, len(test_dataset), args.batch_size):
            cur_dataset = test_dataset[i: i + args.batch_size]
            current_batch = from_example_list(args, cur_dataset, device, train=False)
            pred = model.decode(Example.label_vocab, current_batch)
            for pi, p in enumerate(pred):
                did = current_batch.did[pi]
                predictions[did] = p
    test_json = json.load(open(test_path, 'r', encoding='utf-8'))
    ptr = 0
    for ei, example in enumerate(test_json):
        for ui, utt in enumerate(example):
            utt['pred'] = [pred.split('-') for pred in predictions[f"{ei}-{ui}"]]
            ptr += 1
    json.dump(test_json, open(os.path.join(args.dataroot, 'prediction.json'), 'w',encoding='utf-8'), indent=4, ensure_ascii=False)


if not args.testing:
    num_training_steps = ((len(train_dataset) + args.batch_size - 1) // args.batch_size) * args.max_epoch # 每个epoch都要训练一遍整个dataset
    print('Total training steps: %d' % (num_training_steps))
    optimizer = set_optimizer(model, args)
    #此处添加lr scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    nsamples, best_result = len(train_dataset), {'dev_acc': 0., 'dev_f1': 0.} # nsample:数据集大小 best_result:开发集上的最佳性能指标
    train_index, step_size = np.arange(nsamples), args.batch_size

    if not os.path.exists(os.path.join('weight',args.name,'checkpoint')):
        os.makedirs(os.path.join('weight',args.name,'checkpoint'),exist_ok=True)
    with open(os.path.join('weight',args.name,'config.json'),'w') as f:
        json.dump(args.__dict__, f, indent=4)  # 将 args.__dict__ 的内容以 JSON 格式写入config.json
    logging.basicConfig(filename=os.path.join('weight',args.name,'log.txt'), level=logging.INFO)
    # 设置日志记录的级别为 INFO。这意味着所有 INFO 级别以上的日志（包括 WARNING, ERROR 等）都会被记录到上述指定的文件中。
    
    print('Start training ......')
    for i in range(args.max_epoch):
        start_time = time.time()
        epoch_loss = 0
        np.random.shuffle(train_index) #数据随机打乱
        model.train() # 模型设置为训练模式
        count = 0
        for j in range(0, nsamples, step_size):
            if args.train_manual and count % args.train_manual_c == 0: #每args.train_manual_c个batch用1次
                cur_dataset = [train_manual_dataset[k] for k in train_index[j: j + step_size]]
            else:
                cur_dataset = [train_dataset[k] for k in train_index[j: j + step_size]]
            current_batch = from_example_list(args, cur_dataset, device, train=True)
            output, loss = model(current_batch) # 执行模型的前向传播，并计算损失
            epoch_loss += loss.item()
            loss.backward() # 执行反向传播算法，计算损失函数关于模型参数的梯度。
            optimizer.step() # 更新模型参数，基于计算得到的梯度和定义的优化算法
            optimizer.zero_grad() # 梯度是累积的，因此每次迭代后需要清零
            count += 1
        print('Training: \tEpoch: %d\tTime: %.4f\tTraining Loss: %.4f' % (i, time.time() - start_time, epoch_loss / count))
        logging.info('Training: \tEpoch: %d\tTime: %.4f\tTraining Loss: %.4f' % (i, time.time() - start_time, epoch_loss / count)) #日志中记录训练信息
        torch.cuda.empty_cache()
        gc.collect()

        # if args.train_manual:
        #     start_time = time.time()
        #     metrics, dev_loss = decode('dev', manual=True)
        #     dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
        #     print('Evaluation using Manual transcript: \tEpoch: %d\tTime: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, time.time() - start_time, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
        #     logging.info('Evaluation using Manual transcript: \tEpoch: %d\tTime: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, time.time() - start_time, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
        
        start_time = time.time()
        metrics, dev_loss = decode('dev')
        dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
        print('Evaluation: \tEpoch: %d\tTime: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, time.time() - start_time, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
        logging.info('Evaluation: \tEpoch: %d\tTime: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, time.time() - start_time, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
        if dev_acc > best_result['dev_acc']:
            best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1'], best_result['iter'] = dev_loss, dev_acc, dev_fscore, i
            model_path = os.path.join('weight', args.name, 'checkpoint', 'model.bin')
            torch.save({
                'epoch': i, 'model': model.state_dict(),
                'optim': optimizer.state_dict(),
            }, open(model_path, 'wb'))
            print('NEW BEST MODEL: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
            logging.info('NEW BEST MODEL: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
    print('FINAL BEST RESULT: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.4f\tDev fscore(p/r/f): (%.4f/%.4f/%.4f)' % (best_result['iter'], best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1']['precision'], best_result['dev_f1']['recall'], best_result['dev_f1']['fscore']))
    logging.info('FINAL BEST RESULT: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.4f\tDev fscore(p/r/f): (%.4f/%.4f/%.4f)' % (best_result['iter'], best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1']['precision'], best_result['dev_f1']['recall'], best_result['dev_f1']['fscore']))

else:
    start_time = time.time()
    metrics, dev_loss = decode('dev')
    dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
    predict()
    print("Evaluation costs %.2fs ; Dev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)" % (time.time() - start_time, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
