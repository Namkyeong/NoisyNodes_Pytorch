import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops
import utils
import os
import os.path as osp
import sys

from embedder import GraphNetwork

from tensorboardX import SummaryWriter

from data import QM9


def test(model, data_loader, criterion, device):
    model.eval()

    with torch.no_grad():
        loss = 0
        for bc, batch in enumerate(data_loader):
            batch.to(device)

            preds, _, _ = model(batch)
            preds = preds[-1]
            loss += torch.sqrt(criterion(batch.y, preds))
            
    return loss/(bc + 1)


def get_pred(model, data_loader, device):
    model.eval()

    temp = list()

    with torch.no_grad():
        for batch in data_loader:
            batch.to(device)
            preds = model(batch).reshape(-1).detach().cpu().numpy()
            y = batch.y.detach().cpu().numpy()
            temp.append([batch.mp_id, preds, y])
                
    return temp


def main():
    
    args = utils.parse_args()
    train_config = utils.training_config(args)
    configuration = utils.get_name(train_config)
    print("{}".format(configuration))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"

    # Tensorboard writer
    WRITER_PATH = "runs/"
    log_dir = WRITER_PATH + configuration
    writer = SummaryWriter(log_dir=log_dir)

    # Model Checkpoint Path
    CHECKPOINT_PATH = "model_checkpoints/"
    check_dir = CHECKPOINT_PATH + configuration + ".pt"

    target = args.target
    targets = ['mu (D)', 'a (a^3_0)', 'e_HOMO (eV)', 'e_LUMO (eV)', 'delta e (eV)', 'R^2 (a^2_0)', 'ZPVE (eV)', 'U_0 (eV)', 'U (eV)', 'H (eV)', 'G (eV)', 'c_v (cal/mol.K)', ]

    # Load dataset
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'QM9')
    transform = utils.MyTransform(target, args.hidden_dim)
    dataset = QM9(path, transform = transform).shuffle()
    print('Dataset Loaded! --> # of graphs:', len(dataset))
    print('Loaded the QM9 dataset. Target property: ', targets[args.target])
    
    # Split dataset
    train_dataset = dataset[:110000]
    val_dataset = dataset[110000:120000]
    test_dataset = dataset[120000:]

    #Load dataset
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Parameter Settings
    n_atom_feat = train_dataset[0].x.shape[1]
    n_bond_feat = train_dataset[0].edge_attr.shape[1]
    hidden_dim = args.hidden_dim
    dim_out = 1

    # Model selection
    model = GraphNetwork(args.M, args.N, n_atom_feat, hidden_dim + 3, hidden_dim, device).to(device)
    print(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    criterion_node = nn.MSELoss()

    train_loss = 0
    best_val_loss = 1000
    count = 0
    num_batch = int(len(train_dataset)/args.batch_size)
    best_losses = list()

    f = open("experiments.txt", "a")
    test_loss = test(model, test_loader, criterion, device)
    print("\n {} epochs --> test_loss : {:.4f} \n".format(0, test_loss))
    writer.add_scalar("loss/test_loss", test_loss, 0)

    preprocessing = utils.Noise_injection(std = args.noise_std)
    
    for epoch in range(args.epochs):

        model.train()

        for bc, batch in enumerate(train_loader):
            
            batch.to(device)
            batch = preprocessing._preprocess(batch, hidden_dim, device)
            
            preds, x_reconstruction, x_list = model(batch)
            loss = criterion(batch.y.reshape(1, -1).expand(args.N, -1), preds) \
                    + args.alpha * criterion_node(x_reconstruction, batch.noise.reshape(1, batch.noise.shape[0], batch.noise.shape[1]).expand(args.N, -1, -1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += torch.sqrt(loss)
            count += 1

            sys.stdout.write('\repoch {}/{} batch {}/{} --> train_loss : {:.4f}'.format(epoch + 1, args.epochs, bc + 1, num_batch + 1, train_loss/count))
            sys.stdout.flush()
        
        writer.add_scalar("loss/train_loss", train_loss/count, epoch)

        if (epoch + 1) % 5 == 0 :
            
            MADs = list()
            for i in range(x_list.shape[0]):
                MAD = torch.cdist(x_list[i], x_list[i], p = 1).mean()
                MADs.append(MAD)
                writer.add_scalar("loss/MAD_block{}".format(i), MAD, epoch)

            val_loss = test(model, val_loader, criterion, device)
            test_loss = test(model, test_loader, criterion, device)
            print("\n {} epochs --> val_loss: {:.4f} test_loss: {:.4f} \n".format(epoch + 1, val_loss, test_loss))
            print("MADs per layers --> {} \n".format(MADs))
            writer.add_scalar("loss/val_loss", val_loss, epoch + 1)
            writer.add_scalar("loss/test_loss", test_loss, epoch + 1)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_epoch = epoch + 1
            
            best_losses.append(best_test_loss)
            st_best = '** [Best epoch: {}] Best test: {:.4f} **\n'.format(best_epoch, best_test_loss)
            print(st_best)

            if len(best_losses) > int(args.es / 5):
                if best_losses[-1] == best_losses[-int(args.es / 5)]:
                    
                    print("Early stop!!")
                    print("[Final] {}".format(st_best))
                    
                    f.write("\n")
                    f.write("Early stop!!\n")
                    f.write(configuration)
                    f.write("\nbest epoch : {} \n".format(best_epoch))
                    f.write("best MSE : {} \n".format(best_test_loss))
                    
                    sys.exit()
    
    print("\ntraining done!")
    # get predictions and true sequence

    # write experimental results
    f.write("\n")
    f.write(configuration)
    f.write("\nbest epoch : {} \n".format(best_epoch))
    f.write("best val MSE : {:.4f} | best test MSE : {:.4f} \n".format(best_val_loss, best_test_loss))
    f.close()


if __name__ == "__main__" :
    main()