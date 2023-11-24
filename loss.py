class GramMatrix(nn.Module):
    def forwar(self, input):
        b,c,h,w = input.size()
        F = input.view(b,c,h*w)
        G = torch.bmm(F,F.transpose(1,2))
        G.div_(h*w)
        return G
    
class GramMSELoss(nn.Module):
    def forward(self,input,target):
        out = nn.MSELoss()(GramMatrix()(input),target)
        return(out)
    

def transfer():
        
    #define layers, loss functions, weights and compute optimization targets
    style_layers = ['r11','r21','r31','r41', 'r51'] 
    content_layers = ['r42']
    loss_layers = style_layers + content_layers
    loss_fns = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
    if torch.cuda.is_available():
        loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]
        
    #these are good weights settings:
    style_weights = [1e3/n**2 for n in [64,128,256,512,512]]
    content_weights = [1e0]
    weights = style_weights + content_weights

    #compute optimization targets
    style_targets = [GramMatrix()(A).detach() for A in vgg(style_image, style_layers)]
    content_targets = [A.detach() for A in vgg(content_image, content_layers)]
    targets = style_targets + content_targets

    #run style transfer
    max_iter = 500
    show_iter = 50
    optimizer = optim.LBFGS([opt_img]);
    n_iter=[0]

    while n_iter[0] <= max_iter:

        def closure():
            optimizer.zero_grad()
            out = vgg(opt_img, loss_layers)
            layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a,A in enumerate(out)]
            loss = sum(layer_losses)
            loss.backward()
            n_iter[0]+=1
            #print loss
            if n_iter[0]%show_iter == (show_iter-1):
                print('Iteration: %d, loss: %f'%(n_iter[0]+1, loss.item()))
    #             print([loss_layers[li] + ': ' +  str(l.data[0]) for li,l in enumerate(layer_losses)]) #loss of each layer
            return loss
        
        optimizer.step(closure)
        
    #display result
    out_img = postp(opt_img.data[0].cpu().squeeze())
    imshow(out_img)
    gcf().set_size_inches(10,10)