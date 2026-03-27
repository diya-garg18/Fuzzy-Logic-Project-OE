% Student Performance Prediction using ANFIS

data = [
80 75 70 3
60 65 60 2
40 50 45 1
90 85 88 3
30 40 35 1
70 72 68 2
85 80 82 3
50 55 52 2
35 30 40 1
75 78 74 3
];

% Generate initial FIS
fis = genfis1(data,3,'gbellmf');

% Train ANFIS model
[trainedFis, error] = anfis(data,fis,50);

% Plot training error
figure
plot(error)
title('Training Error')
xlabel('Epochs')
ylabel('Error')