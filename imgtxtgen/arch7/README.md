Different variations of a simple image captioning network architecture.

1. CNN + RNN trained with MLE.
2. CNN + RNN trained with MLE and using beam search for evaluation.

3. CNN + RNN + soft_attention(hiddens, image_features) trained with MLE.
4. CNN + RNN + soft_attention(hiddens, image_features) trained with MLE and using beam search for evaluation.

5. CNN + RNN + soft_attention(hiddens, prev_word_embeddings, image_features) trained with MLE.
6. CNN + RNN + soft_attention(hiddens, prev_word_embeddings, image_features) trained with MLE and using beam search for evaluation.

7. CNN + RNN + discriminator trained with RL to optimize expected reward.
8. CNN + RNN + discriminator trained with RL to optimize expected reward and using beam search for evaluation.

9. CNN + RNN + discriminator trained with RL to optimize best reward.
10. CNN + RNN + discriminator trained with RL to optimize best reward and using beam search for evaluation.

Also for every variation try using beam search during training as well.
