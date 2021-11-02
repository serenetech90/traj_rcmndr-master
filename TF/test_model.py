import tensorflow as tf
from strggrnn_model import strggrnn as model
import json

with open('Sirius_json/test_agents.json') as f:
    test = json.load(f)

results = {k: {} for k in test.keys()}

data = TrajectoryDataset('Sirius_json/', 'test')
data_loader = DataLoader(
    data,
    batch_size=64,
    num_workers=2,
    collate_fn=torch_collate_fn,
    shuffle=False,
)

print('Predicting and preparation of the submission file')

model..load_state_dict(torch.load('best.ckpt', map_location='cpu'))
CNN_Model.eval()
with torch.no_grad():
    for batch in tqdm(data_loader):
        batch = to_gpu(batch)
        preds = CNN_Model(batch)
        batch['coords'][:] = preds[:, NUM_OF_PREDS - 20:]
        preds2 = CNN_Model(batch)
        preds = torch.cat((preds, preds2), 1)

        preds_short = preds[:, :20]
        preds_medium = preds[:, :40:2]
        preds_long = preds[:, :80:4]

        for i, (scene, agent) in enumerate(zip(batch['scene_id'][:, 0], batch['track_id'][:, 0])):
            scene = int(scene.item())
            agent = int(agent.item())
            if agent in test[str(scene)]:
                results[str(scene)][agent] = {}
                results[str(scene)][agent]['short'] = preds_short[i].tolist()
                results[str(scene)][agent]['medium'] = preds_medium[i].tolist()
                results[str(scene)][agent]['long'] = preds_long[i].tolist()

with open('CNN_Submit.json', 'w') as f:
    json.dump(results, f)