# Things to Check

- [ ] cfg free guidance. it was messed-with during debugging. make sure it works like the reference implementation from upstream.
- [ ] check if gradient clipping is working. gradients are huge, even with it turned on. also learn whether gradients need to be scaled based on batch size.
- [ ] the samples from the currently training checkpoint look like they're made up of tiny images tiled together. is this an attention bug, or just a symptom of insufficient training?
    ![Sample from epoch 30](training_samples/116-14800-media_images_samples_14726_64d685c198ed21a554b5.png)
- [ ] curriculum change causes loss spike and gradient instability. in a test run, loss jumped from ~9.6 to ~10.6 at epoch 20, and hadn't started dropping again at epoch 31.
- [ ] compare to upstream to check for drift in implementation.
- [ ] design an optimiser that's a neural network?
