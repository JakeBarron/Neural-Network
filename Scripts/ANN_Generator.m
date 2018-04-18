%/////////////ANN Structure generator///////////////////////////////////////////
% Published by Jake Barron
% This script generates 5 ANNs that take in inputs with 4 features and classify
% them into 3 classes.  L1-L5 have 1,3,5,6,7 layers respectively.

L1 = [4 randi([2 20], 1, 1) 3];
L2 = [4 randi([2 20], 1, 1), randi([2 20], 1, 1), randi([2 20], 1, 1) 3];
L3 = [4 randi([2 20], 1, 1), randi([2 20], 1, 1), randi([2 20], 1, 1), randi([2 20], 1, 1), randi([2 20], 1, 1), 3];
L4 = [4 randi([2 20], 1, 1), randi([2 20], 1, 1), randi([2 20], 1, 1), randi([2 20], 1, 1), randi([2 20], 1, 1), randi([2 20], 1, 1) 3];
L5 = [4 randi([2 20], 1, 1), randi([2 20], 1, 1), randi([2 20], 1, 1), randi([2 20], 1, 1), randi([2 20], 1, 1), randi([2 20], 1, 1), randi([2 20], 1, 1), 3];

ANNs = {L1, L2, L3, L4, L5};

save('ANNs.mat', 'L1', 'L2', 'L3', 'L4', 'L5');
