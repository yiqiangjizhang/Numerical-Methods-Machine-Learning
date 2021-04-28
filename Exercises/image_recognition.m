function ImageRecWebCam
% Set interpreter to latex
set(groot,'defaultAxesTickLabelInterpreter','latex');  
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

camera = webcam();
nnet = googlenet;
while true
    picture = camera.snapshot;
    image(picture)
    picture = imresize(picture,[224,224]);
    [label,score] = classify(nnet,picture);
    title({char(label),num2str(max(score),2)});
    grid on
    grid minor
    box on
    drawnow;
end
end