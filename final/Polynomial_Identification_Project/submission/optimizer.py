polyNet3 = PolyNet().to(device)
polyNet3.apply(init_weights)
optimizer3= torch.optim.RMSprop(polyNet3.parameters(), lr=0.01, alpha=0.99, weight_decay=0.05)
exp_lr_scheduler3 = lr_scheduler.StepLR(optimizer3,step_size=4, gamma=0.8)
