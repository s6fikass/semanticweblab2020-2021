import os
import time

import torch
from torch.utils.data import DataLoader

from pytorch.utils import load_args, get_optimizer, make_out_dir
from pytorch.data import DataModel, TrainDataset, TestDataset
from pytorch.model import MultiKENet
from pytorch.loss import MultiKELoss


def run_itc(args):
    data = DataModel(args)

    views = ['rv', 'ckgrtv', 'ckgrrv', 'av', 'ckgatv', 'ckgarv', 'cnv']

    batch_sizes = [args.batch_size] * len(views)
    batch_sizes[3:6] = [args.attribute_batch_size] * 3
    batch_sizes[-1] = args.entity_batch_size
    train_datasets = [TrainDataset(data, bs, v) for bs, v in zip(batch_sizes, views)]
    train_datasets[0].num_neg_triples = args.num_neg_triples
    train_dataloaders = [DataLoader(ds, bs, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
                         if len(ds) > 0 else ds for ds, bs in zip(train_datasets, batch_sizes)]
    valid_dataset = TestDataset(data.kgs.get_entities('valid', 1), data.kgs.get_entities('validtest', 2))
    test_dataset = TestDataset(data.kgs.get_entities('test', 1), data.kgs.get_entities('test', 2))
    valid_dataloader = DataLoader(valid_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiKENet(data.kgs.num_entities, data.kgs.num_relations, data.kgs.num_attributes, args.embed_dim, data.value_vectors, data.local_name_vectors, args.mode, args.num_vectors)
    model.to(device)

    lrs = [args.learning_rate] * len(views)
    lrs[-1] = args.itc_learning_rate
    criterion = MultiKELoss(args.cv_name_weight, args.cv_weight)
    optimizers = [get_optimizer(args.optimizer, model.parameters(v), lr) for v, lr in zip(views, lrs)]

    out_path = make_out_dir(args.output, args.dataset, args.dataset_division, 'MultiKECV')
    pre_value, two_pre_value = -1, -1

    model.test(args, model, test_dataloader, embed_choice='nv')

    for i in range(1, args.epochs + 1):
        print("epoch {}:".format(i))

        for idx, view in enumerate(views):
            train_dataloader = train_dataloaders[idx]
            if len(train_dataloader) == 0 or (view in ['ckgrrv', 'ckgarv'] and i <= args.start_predicate_soft_alignment):
                continue

            total = 0
            running_loss = 0.0
            optimizer = optimizers[idx]
            model.train()
            start_time = time.time()
            for inputs, weights in train_dataloader:
                if view == 'rv':
                    inputs_pos = list(map(lambda x: x.to(device), inputs[:3]))
                    inputs_negs = list(map(lambda x: torch.cat(x, dim=0).to(device), inputs[3:]))
                    inputs = inputs_pos + inputs_negs
                else:
                    inputs = list(map(lambda x: x.long().to(device), inputs))
                weights = list(map(lambda x: x.float().to(device), weights))

                optimizer.zero_grad()
                outputs = model(inputs, view)
                loss = criterion(outputs, weights, view)
                loss.backward()
                optimizer.step()

                total += inputs[0].size(0)
                running_loss += loss.item() * inputs[0].size(0)

            end_time = time.time()
            print("epoch {} of {}, avg. loss: {:.4f}, time: {:.4f}s".format(i, view, running_loss / total, end_time - start_time))

        if i >= args.start_valid and i % args.eval_freq == 0:
            model.test(args, model, valid_dataloader, embed_choice='rv')
            model.test(args, model, valid_dataloader, embed_choice='av')
            metric_value = model.test(args, model, valid_dataloader, embed_choice='final')

            early_stop = metric_value <= pre_value <= two_pre_value
            two_pre_value, pre_value = pre_value, metric_value
            if early_stop or i == args.epochs:
                break

            if i >= args.start_predicate_soft_alignment and i % 10 == 0:
                data.update_predicate_alignment(model)

        if args.neg_sampling == 'truncated' and i % args.truncated_freq == 0:
            assert 0.0 < args.truncated_epsilon < 1.0
            data.generate_neighbours(model, args.truncated_epsilon)

        for ds in train_datasets:
            ds.regenerate()

    # save checkpoint
    torch.save({
        'epoch': i,
        'model': model.state_dict(),
        'optimizers': [opt.state_dict() for opt in optimizers]
    }, os.path.join(out_path, 'checkpoint.pth'))

    model.test(args, model, test_dataloader, embed_choice='nv', accurate=True)
    model.test(args, model, test_dataloader, embed_choice='rv', accurate=True)
    model.test(args, model, test_dataloader, embed_choice='av', accurate=True)
    model.test(args, model, test_dataloader, embed_choice='final', accurate=True)


def run_ssl(args):
    data = DataModel(args)

    views = ['rv', 'ckgrtv', 'ckgrrv', 'av', 'ckgatv', 'ckgarv', 'mv']

    batch_sizes = [args.batch_size] * len(views)
    batch_sizes[3:6] = [args.attribute_batch_size] * 3
    batch_sizes[-1] = args.entity_batch_size
    train_datasets = [TrainDataset(data, bs, v) for bs, v in zip(batch_sizes, views)]
    train_datasets[0].num_neg_triples = args.num_neg_triples
    train_dataloaders = [DataLoader(ds, bs, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory) for ds, bs in zip(train_datasets, batch_sizes)]
    valid_dataset = TestDataset(data.kgs.get_entities('valid', 1), data.kgs.get_entities('validtest', 2))
    test_dataset = TestDataset(data.kgs.get_entities('test', 1), data.kgs.get_entities('test', 2))
    valid_dataloader = DataLoader(valid_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiKENet(data.kgs.num_entities, data.kgs.num_relations, data.kgs.num_attributes, args.embed_dim, data.value_vectors, data.local_name_vectors, args.mode, args.num_vectors, True)
    model.to(device)

    lrs = [args.learning_rate] * len(views)
    criterion = MultiKELoss(args.cv_name_weight, args.cv_weight, args.orthogonal_weight, model.eye)
    optimizers = [get_optimizer(args.optimizer, model.parameters(v), lr) for v, lr in zip(views, lrs)]

    out_path = make_out_dir(args.output, args.dataset, args.dataset_division, 'MultiKELate')
    pre_value, two_pre_value = -1, -1

    model.test(args, model, test_dataloader, embed_choice='nv')
    # model.test(args, model, test_dataloader, embed_choice='avg')

    for i in range(1, args.epochs + 1):
        print("epoch {}:".format(i))

        for idx, view in enumerate(views[:-1]):
            train_dataloader = train_dataloaders[idx]
            if len(train_dataloader) == 0 or (view in ['ckgrrv', 'ckgarv'] and i <= args.start_predicate_soft_alignment):
                continue

            total = 0
            running_loss = 0.0
            optimizer = optimizers[idx]
            model.train()
            start_time = time.time()
            for inputs, weights in train_dataloader:
                if view == 'rv':
                    inputs_pos = list(map(lambda x: x.to(device), inputs[:3]))
                    inputs_negs = list(map(lambda x: torch.cat(x, dim=0).to(device), inputs[3:]))
                    inputs = inputs_pos + inputs_negs
                else:
                    inputs = list(map(lambda x: x.long().to(device), inputs))
                weights = list(map(lambda x: x.float().to(device), weights))

                optimizer.zero_grad()
                outputs = model(inputs, view)
                loss = criterion(outputs, weights, view)
                loss.backward()
                optimizer.step()

                total += inputs[0].size(0)
                running_loss += loss.item() * inputs[0].size(0)

            end_time = time.time()
            print("epoch {} of {}, avg. loss: {:.4f}, time: {:.4f}s".format(i, view, running_loss / total, end_time - start_time))

        # save checkpoint
        torch.save({
            'epoch': i,
            'model': model.state_dict(),
            'optimizers': [opt.state_dict() for opt in optimizers]
        }, os.path.join(out_path, 'checkpoint.pth'))

        if i >= args.start_valid and i % args.eval_freq == 0:
            model.test(args, model, valid_dataloader, embed_choice='rv')
            model.test(args, model, valid_dataloader, embed_choice='av')
            # model.test(args, model, valid_dataloader, embed_choice='avg')
            metric_value = model.test_wva(args, model, valid_dataloader)

            early_stop = metric_value <= pre_value <= two_pre_value
            two_pre_value, pre_value = pre_value, metric_value
            if early_stop or i == args.epochs:
                break

            if i >= args.start_predicate_soft_alignment:
                data.update_predicate_alignment(model)

        if args.neg_sampling == 'truncated' and i % args.truncated_freq == 0:
            assert 0.0 < args.truncated_epsilon < 1.0
            data.generate_neighbours(model, args.truncated_epsilon)

        for ds in train_datasets[:-1]:
            ds.regenerate()

    view = views[-1]
    optimizer = optimizers[-1]
    train_dataloader = train_dataloaders[-1]
    train_dataset = train_datasets[-1]
    for i in range(1, args.shared_learning_epochs + 1):
        total = 0
        running_loss = 0.0
        model.train()
        start_time = time.time()
        for inputs, weights in train_dataloader:
            inputs = list(map(lambda x: x.long().to(device), inputs))
            weights = list(map(lambda x: x.float().to(device), weights))

            optimizer.zero_grad()
            outputs = model(inputs, view)
            loss = criterion(outputs, weights, view)
            loss.backward()
            optimizer.step()

            total += inputs[0].size(0)
            running_loss += loss.item() * inputs[0].size(0)

        end_time = time.time()
        print("epoch {} of {}, avg. loss: {:.4f}, time: {:.4f}s".format(i, view, running_loss / total, end_time - start_time))

        if i >= args.start_valid and i % args.eval_freq == 0:
            model.test(args, model, valid_dataloader, embed_choice='final')

        train_dataset.regenerate()

    model.test(args, model, test_dataloader, embed_choice='nv', accurate=True)
    model.test(args, model, test_dataloader, embed_choice='rv', accurate=True)
    model.test(args, model, test_dataloader, embed_choice='av', accurate=True)
    # model.test(args, model, test_dataloader, embed_choice='avg', accurate=True)
    model.test_wva(args, model, test_dataloader, accurate=True)
    model.test(args, model, test_dataloader, embed_choice='final', accurate=True)
