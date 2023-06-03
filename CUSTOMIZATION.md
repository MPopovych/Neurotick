# Intro
The library supports definition of own custom layers and activation functions. 
Several requirements need to be met:

- Structures are abstract enough that they do not depend on unserializable or runtime references
- They can be serialized into a JsonWrap (wrap over serde's Value)  via the trait methods required for activation and layers
- They can implement instantiation from a JSON

Deserialization is done by providing a **GenericInjector** into the model reader. Example:

``` rust
let model_pre_serialize = build_small_model();
let serialized: ModelSerialized = model_pre_serialize.to_serialized_model();
/* ... */
let model_reader = ModelReader::default();
let model_post_serialize = serialized.build_model(&model_reader);
```

The model reader registers the parsing logic per implementation by name. Thus the name should be unique for every implementation.

The default function provides the library implementations + allows registering your own implementations by calling ModelReader::register on a mutable reference;