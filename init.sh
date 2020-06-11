## Creadte checkpoints folder
mkdir -p checkpoints

## Donwnload dcgan models
wget https://github.com/NTU-speech-lab/hw11-TsengMJ/releases/download/0/p1_d.pth -P ./checkpoints
wget https://github.com/NTU-speech-lab/hw11-TsengMJ/releases/download/0/p1_g.pth -P ./checkpoints

## Donwnload wgan models
wget https://github.com/NTU-speech-lab/hw11-TsengMJ/releases/download/1/p2_d.pth -P ./checkpoints
wget https://github.com/NTU-speech-lab/hw11-TsengMJ/releases/download/1/p2_g.pth -P ./checkpoints
