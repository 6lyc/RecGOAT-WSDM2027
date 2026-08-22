from parse import parse_args
import world
from register import dataset_factory, model_factory
from utils import Loss
import Procedure

if __name__ == '__main__':
    args = parse_args()
    world.config = world.Config(args)
    world.set_seed(world.config.seed)
    world.cprint(f"Config:\n{world.config.as_dict()}")

    dataset = dataset_factory()
    Recmodel = model_factory(dataset)
    loss = Loss(world.config, Recmodel)

    best = 0.0
    for epoch in range(world.config.epochs):
        output_information = Procedure.train_v2(dataset, Recmodel, loss, epoch)
        if (epoch+1) % 20 == 0:
            results = Procedure.Test(dataset, Recmodel, epoch, multicore=world.config.multicore)
            
            score = results['ndcg'][0]
            if score > best:
                best = score
                print(f"Better@{epoch}]", {k: v for k,v in results.items()})

        
        print(f"Epoch {epoch:03d}: {output_information}")