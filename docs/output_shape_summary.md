# Output Shape Summary

## Cache Sample
- svg_tensor shape: [3, 14]
- first row: all zeros (likely BOS/PAD or very short sample)
- pixel_cls shape: [384]
- caption: None (cache created before caption feature)

## VAE Output
- predicted_features shape: [3, 14]
- value range: roughly [-1, 1] (tanh output)
- row0 sample (first 6 values): [0.25, -0.45, -0.99, ...]

## DiT Output
- pred_noise shape: [1, 1024, 128]
- values are unconstrained (noise prediction)

## Files Generated
- outputs/dit_sample_*.svg
- outputs/dit_sample_*.svg.png
- outputs/recon/*
- outputs/samples/*

## Notes
- Captions were None because the cache was generated before caption mapping was added.
- To include captions, rerun preprocess after updating code.
