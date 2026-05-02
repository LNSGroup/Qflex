from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Sequence, Tuple, Union
import numpy as np
import jax, jax.numpy as jnp
import haiku as hk
from haiku.initializers import Constant
from relax.utils.jax_utils import fix_repr, is_broadcastable


Activation = Callable[[jax.Array], jax.Array]
Identity: Activation = lambda x: x
Tanh: Activation = lambda x: jnp.tanh(x)


@dataclass
@fix_repr
class BatchRenorm(hk.BatchNorm):
    def __init__(
      self,
      create_scale: bool,
      create_offset: bool,
      decay_rate: float,
      eps: float = 1e-3,
      scale_init: hk.initializers.Initializer | None = None,
      offset_init: hk.initializers.Initializer | None = None,
      axis: Sequence[int] | None = None,
      cross_replica_axis: str | Sequence[str] | None = None,
      cross_replica_axis_index_groups: Sequence[Sequence[int]] | None = None,
      data_format: str = "channels_last",
      name: str | None = None,
  ):
        """Constructs a BatchNorm module.

        Args:
        create_scale: Whether to include a trainable scaling factor.
        create_offset: Whether to include a trainable offset.
        decay_rate: Decay rate for EMA.
        eps: Small epsilon to avoid division by zero variance. Defaults ``1e-5``,
            as in the paper and Sonnet.
        scale_init: Optional initializer for gain (aka scale). Can only be set
            if ``create_scale=True``. By default, ``1``.
        offset_init: Optional initializer for bias (aka offset). Can only be set
            if ``create_offset=True``. By default, ``0``.
        axis: Which axes to reduce over. The default (``None``) signifies that all
            but the channel axis should be normalized. Otherwise this is a list of
            axis indices which will have normalization statistics calculated.
        cross_replica_axis: If not ``None``, it should be a string (or sequence of
            strings) representing the axis name(s) over which this module is being
            run within a jax map (e.g. ``jax.pmap`` or ``jax.vmap``). Supplying this
            argument means that batch statistics are calculated across all replicas
            on the named axes.
        cross_replica_axis_index_groups: Specifies how devices are grouped. Valid
            only within ``jax.pmap`` collectives.
        data_format: The data format of the input. Can be either
            ``channels_first``, ``channels_last``, ``N...C`` or ``NC...``. By
            default it is ``channels_last``. See :func:`get_channel_index`.
        name: The module name.
        """
        super().__init__(
            create_scale=create_scale,
            create_offset=create_offset,
            decay_rate=decay_rate,
            eps=eps, 
            scale_init=scale_init, 
            offset_init=offset_init, 
            axis=axis, 
            cross_replica_axis=cross_replica_axis, 
            cross_replica_axis_index_groups=cross_replica_axis_index_groups, 
            data_format=data_format, 
            name=name
        )
        self.r_max = jnp.array([3]).astype(jnp.float32)
        self.d_max = jnp.array([5]).astype(jnp.float32)
        self.counter = jnp.array([0]).astype(jnp.float32)
        self.warmup_th = jnp.array([10]).astype(jnp.float32)

    def __call__(
      self,
      inputs: jax.Array,
      is_training: bool,
      test_local_stats: bool = False,
      scale: jax.Array | None = None,
      offset: jax.Array | None = None,
    ) -> jax.Array:
        """Computes the normalized version of the input.

        Args:
        inputs: An array, where the data format is ``[..., C]``.
        is_training: Whether this is during training.
        test_local_stats: Whether local stats are used when is_training=False.
        scale: An array up to n-D. The shape of this tensor must be broadcastable
            to the shape of ``inputs``. This is the scale applied to the normalized
            inputs. This cannot be passed in if the module was constructed with
            ``create_scale=True``.
        offset: An array up to n-D. The shape of this tensor must be broadcastable
            to the shape of ``inputs``. This is the offset applied to the normalized
            inputs. This cannot be passed in if the module was constructed with
            ``create_offset=True``.

        Returns:
        The array, normalized across all but the last dimension.
        """
        if self.create_scale and scale is not None:
            raise ValueError(
                "Cannot pass `scale` at call time if `create_scale=True`.")
        if self.create_offset and offset is not None:
            raise ValueError(
                "Cannot pass `offset` at call time if `create_offset=True`.")

        channel_index = self.channel_index
        if channel_index < 0:
            channel_index += inputs.ndim

        if self.axis is not None:
            axis = self.axis
        else:
            axis = [i for i in range(inputs.ndim) if i != channel_index]

        if is_training or test_local_stats:
            mean = jnp.mean(inputs, axis, keepdims=True)
            mean_of_squares = jnp.mean(jnp.square(inputs), axis, keepdims=True)
            if self.cross_replica_axis:
                mean = jax.lax.pmean(
                    mean,
                    axis_name=self.cross_replica_axis,
                    axis_index_groups=self.cross_replica_axis_index_groups)
                mean_of_squares = jax.lax.pmean(
                    mean_of_squares,
                    axis_name=self.cross_replica_axis,
                    axis_index_groups=self.cross_replica_axis_index_groups)
            var = mean_of_squares - jnp.square(mean)
            var = jnp.maximum(var, self.eps)
            custom_mean = mean
            custom_var = var
            r = 1
            d = 0
            std = jnp.sqrt(jnp.maximum(self.eps, var + self.eps))
            try:
                var_ema = self.var_ema.average.astype(inputs.dtype)
                mean_ema = self.mean_ema.average.astype(inputs.dtype)
            except Exception as e: # not initialization
                print('Not initialization for EMA, use temp values', e)
                var_ema = jnp.zeros_like(var)
                mean_ema = jnp.zeros_like(mean)
                # counter = hk.get_state("bn_counter", init=lambda: jnp.array([0], dtype=jnp.float32)
                #                        ).astype(jnp.float32)
            counter = hk.get_state("bn_counter", (), jnp.int32,
                           init=hk.initializers.Constant(0))
            # jax.debug.print(f'{counter}')
            ra_std = jnp.sqrt(jnp.maximum(self.eps, var_ema + self.eps))
            r = jax.lax.stop_gradient(std / ra_std)
            r = jnp.clip(r, 1 / self.r_max, self.r_max)
            d = jax.lax.stop_gradient((mean - mean_ema) / ra_std)
            d = jnp.clip(d, -self.d_max, self.d_max)
            tmp_var = var / (r**2)
            tmp_mean = mean - d * jnp.sqrt(custom_var) / (r)
            
            warmed_up = jnp.greater_equal(counter, self.warmup_th).astype(jnp.float32)
            # if warmed_up.astype(jnp.bool):
            #     print('Warmed up BN')
            # print(r.max(), r.min(), d.max(), d.min())
            custom_var = warmed_up * tmp_var + (1. - warmed_up) * custom_var
            custom_mean = warmed_up * tmp_mean + (1. - warmed_up) * custom_mean
            # print('is_training')
        else:
            mean = self.mean_ema.average.astype(inputs.dtype)
            var = self.var_ema.average.astype(inputs.dtype)
            custom_mean = mean
            custom_var = var

        if is_training:
            # print('is_training')
            self.mean_ema(mean)
            self.var_ema(var)
            counter += 1
            hk.set_state("bn_counter", counter)
            
            

        w_shape = [1 if i in axis else inputs.shape[i] for i in range(inputs.ndim)]
        w_dtype = inputs.dtype

        if self.create_scale:
            scale = hk.get_parameter("scale", w_shape, w_dtype, self.scale_init)
        elif scale is None:
            scale = np.ones([], dtype=w_dtype)

        if self.create_offset:
            offset = hk.get_parameter("offset", w_shape, w_dtype, self.offset_init)
        elif offset is None:
            offset = np.zeros([], dtype=w_dtype)

        eps = jax.lax.convert_element_type(self.eps, custom_var.dtype)
        inv = scale * jax.lax.rsqrt(custom_var + eps)
        # print(inputs.shape, custom_mean.shape, inv.shape, offset.shape)
        
        normalized_input = (inputs - custom_mean) * inv + offset
        # print(inputs.shape, custom_mean.shape, inv.shape, offset.shape, normalized_input.shape, jnp.mean(inputs[0]))
        # print(jnp.mean(custom_var), jnp.mean(custom_mean), jnp.mean(inv), jnp.mean(offset), jnp.mean(normalized_input[0]), jnp.mean((inputs[0].reshape(1, -1) - custom_mean) * inv + offset))
        return normalized_input


@dataclass
@fix_repr
class ValueNet(hk.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    name: str = None
    
    def __call__(self, obs: jax.Array) -> jax.Array:
        return mlp(self.hidden_sizes, 1, self.activation, self.output_activation, squeeze_output=True)(obs)


@dataclass
@fix_repr
class QNet(hk.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    name: str = None


    def __call__(self, obs: jax.Array, act: jax.Array) -> jax.Array:
        input = jnp.concatenate((obs, act), axis=-1)
        return mlp(self.hidden_sizes, 1, self.activation, self.output_activation, squeeze_output=True)(input)

@dataclass
@fix_repr
class QNetBN(hk.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    name: str = None

    # @partial(jax.jit, static_argnames=['is_training'])
    def __call__(self, obs: jax.Array, act: jax.Array, is_training: bool = True
                 ) -> jax.Array:
        input = jnp.concatenate((obs, act), axis=-1)
        
        return mlp_with_bn_q(input, self.hidden_sizes, 1, self.activation, self.output_activation, squeeze_output=True, 
                           is_training=is_training
                           )
       


@dataclass
@fix_repr
class DistributionalQNet(hk.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    min_log_std: float = -0.1
    max_log_std: float = 4.0
    name: str = None
    

    def __call__(self, obs: jax.Array, act: jax.Array) -> Tuple[jax.Array, jax.Array]:
        input = jnp.concatenate((obs, act), axis=-1)
        value_mean = mlp(self.hidden_sizes, 1, self.activation, self.output_activation, squeeze_output=True)(input)
        value_log_std = mlp(self.hidden_sizes, 1, self.activation, self.output_activation, squeeze_output=True)(input)
        denominator = max(abs(self.min_log_std), abs(self.max_log_std))
        value_log_std = (
            jnp.maximum( self.max_log_std * jnp.tanh(value_log_std / denominator), 0.0) +
            jnp.minimum(-self.min_log_std * jnp.tanh(value_log_std / denominator), 0.0)
        )
        return value_mean, value_log_std

@dataclass
@fix_repr
class DistributionalQNet2(hk.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    name: str = None

    def __call__(self, obs: jax.Array, act: jax.Array) -> Tuple[jax.Array, jax.Array]:
        input = jnp.concatenate((obs, act), axis=-1)
        output = mlp(self.hidden_sizes, 2, self.activation, self.output_activation)(input)
        value_mean = output[..., 0]
        value_std = jax.nn.softplus(output[..., 1])
        return value_mean, value_std

@dataclass
@fix_repr
class DistributionalQNet2BN(hk.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    name: str = None

    def __call__(self, obs: jax.Array, act: jax.Array, is_training: bool = True) -> Tuple[jax.Array, jax.Array]:
        input = jnp.concatenate((obs, act), axis=-1)
        output = mlp_with_bn_q(input, self.hidden_sizes, 2, self.activation, self.output_activation, 
                           is_training=is_training
                           )
        value_mean = output[..., 0]
        value_std = jax.nn.softplus(output[..., 1])
        return value_mean, value_std

@dataclass
@fix_repr
class PolicyNet(hk.Module):
    act_dim: int
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    min_log_std: float = -20.0
    max_log_std: float = 0.5
    log_std_mode: Union[str, float] = 'shared'  # shared, separate, global (provide initial value)
    name: str = None
    bn: bool = False # Whether to use batch normalization

    
    def __call__(self, obs: jax.Array, *, return_log_std: bool = False) -> jax.Array:
        if self.log_std_mode == 'shared':
            output = mlp(self.hidden_sizes, self.act_dim * 2, self.activation, self.output_activation)(obs)
            mean, log_std = jnp.split(output, 2, axis=-1)
        elif self.log_std_mode == 'separate':
            mean = mlp(self.hidden_sizes, self.act_dim, self.activation, self.output_activation)(obs)
            log_std = mlp(self.hidden_sizes, self.act_dim, self.activation, self.output_activation)(obs)
        else:
            initial_log_std = float(self.log_std_mode)
            mean = mlp(self.hidden_sizes, self.act_dim, self.activation, self.output_activation)(obs)
            log_std = hk.get_parameter('log_std', shape=(self.act_dim,), init=Constant(initial_log_std))
            log_std = jnp.broadcast_to(log_std, mean.shape)
        if not (self.min_log_std is None and self.max_log_std is None):
            log_std = jnp.clip(log_std, self.min_log_std, self.max_log_std)
        if return_log_std:
            return mean, log_std
        else:
            return mean, jnp.exp(log_std)

@dataclass
@fix_repr
class PolicyNetBN(hk.Module):
    act_dim: int
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    min_log_std: float = -20
    max_log_std: float = 2
    log_std_mode: Union[str, float] = 'shared'  # shared, separate, global (provide initial value)
    name: str = None
    bn: bool = False # Whether to use batch normalization

    def __call__(self, obs: jax.Array, *, return_log_std: bool = False, is_training: bool = True
                 ) -> jax.Array:
        if jnp.ndim(obs) == 1:
            obs = obs.reshape(1, -1)
        if self.log_std_mode == 'shared':
            output = mlp_with_bn(obs, self.hidden_sizes, self.act_dim * 2, self.activation, self.output_activation,
                                  is_training=is_training
                                  )
            mean, log_std = jnp.split(output, 2, axis=-1)
        elif self.log_std_mode == 'separate':
            mean = mlp_with_bn(obs, self.hidden_sizes, self.act_dim, self.activation, self.output_activation, 
                               is_training=is_training
                               )
            log_std = mlp_with_bn(obs, self.hidden_sizes, self.act_dim, self.activation, self.output_activation, 
                                  is_training=is_training
                                  )
        else:
            initial_log_std = float(self.log_std_mode)
            mean = mlp_with_bn(obs, self.hidden_sizes, self.act_dim, self.activation, self.output_activation, 
                               is_training=is_training
                               )
            log_std = hk.get_parameter('log_std', shape=(self.act_dim,), init=Constant(initial_log_std))
            log_std = jnp.broadcast_to(log_std, mean.shape)
        if not (self.min_log_std is None and self.max_log_std is None):
            log_std = jnp.clip(log_std, self.min_log_std, self.max_log_std)
        if return_log_std:
            return mean, log_std
        else:
            return mean, jnp.exp(log_std)

@dataclass
@fix_repr
class FlowVelocityField(hk.Module):
    act_dim: int
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    min_log_std: float = -5
    max_log_std: float = 2
    log_std_mode: Union[str, float] = 'shared'  # shared, separate, global (provide initial value)
    name: str = None
    def __call__(self, obs: jax.Array, act: jax.Array, t:jax.Array) -> jax.Array:
        input = jnp.concatenate((obs, act, t), axis=-1)
        return mlp(self.hidden_sizes, self.act_dim, self.activation, self.output_activation, squeeze_output=True)(input)

@dataclass
@fix_repr
class FlowVelocityFieldBN(hk.Module):
    act_dim: int
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    name: str = None
    def __call__(self, obs: jax.Array, act: jax.Array, t:jax.Array, is_training: bool = True) -> jax.Array:
        # print(obs.shape, act.shape, t.shape)
        input = jnp.concatenate((obs, act, t), axis=-1)
        if jnp.ndim(input) == 1:
            input = input.reshape(1, -1)
        # print(input.shape, obs.shape, act.shape, t.shape)
        return mlp_with_bn(input, self.hidden_sizes, self.act_dim, self.activation, self.output_activation, 
                           is_training=is_training
                                  )

@dataclass
@fix_repr
class PolicyStdNet(hk.Module):
    act_dim: int
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Tanh
    min_log_std: float = -5.0
    max_log_std: float = 2.0
    name: str = None

    def __call__(self, obs: jax.Array) -> jax.Array:
        log_std = mlp(self.hidden_sizes, self.act_dim, self.activation, self.output_activation)(obs)
        return self.min_log_std + (log_std + 1) / 2 * (self.max_log_std - self.min_log_std)


@dataclass
@fix_repr
class DeterministicPolicyNet(hk.Module):
    act_dim: int
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    name: str = None

    def __call__(self, obs: jax.Array) -> jax.Array:
        return mlp(self.hidden_sizes, self.act_dim, self.activation, self.output_activation)(obs)


@dataclass
@fix_repr
class ModelNet(hk.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    name: str = None

    def __call__(self, obs: jax.Array, act: jax.Array) -> jax.Array:
        obs_dim = obs.shape[-1]
        input = jnp.concatenate((obs, act), axis=-1)
        return mlp(self.hidden_sizes, obs_dim, self.activation, self.output_activation)(input)


@dataclass
@fix_repr
class QScoreNet(hk.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    name: str = None

    def __call__(self, obs: jax.Array, act: jax.Array) -> jax.Array:
        act_dim = act.shape[-1]
        input = jnp.concatenate((obs, act), axis=-1)
        return mlp(self.hidden_sizes, act_dim, self.activation, self.output_activation)(input)


@dataclass
@fix_repr
class DiffusionPolicyNet(hk.Module):
    time_dim: int
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    name: str = None

    def __call__(self, obs: jax.Array, act: jax.Array, t: jax.Array) -> jax.Array:
        act_dim = act.shape[-1]
        te = scaled_sinusoidal_encoding(t, dim=self.time_dim, batch_shape=obs.shape[:-1])
        input = jnp.concatenate((obs, act, te), axis=-1)
        return mlp(self.hidden_sizes, act_dim, self.activation, self.output_activation)(input)

@dataclass
@fix_repr
class DACERPolicyNet(hk.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    time_dim: int = 16
    name: str = None

    def __call__(self, obs: jax.Array, act: jax.Array, t: jax.Array) -> jax.Array:
        act_dim = act.shape[-1]
        te = scaled_sinusoidal_encoding(t, dim=self.time_dim, batch_shape=obs.shape[:-1])
        te = hk.Linear(self.time_dim * 2)(te)
        te = self.activation(te)
        te = hk.Linear(self.time_dim)(te)
        input = jnp.concatenate((obs, act, te), axis=-1)
        return mlp(self.hidden_sizes, act_dim, self.activation, self.output_activation)(input)

def mlp(hidden_sizes: Sequence[int], output_size: int, activation: Activation, output_activation: Activation, *, squeeze_output: bool = False, 
        ) -> Callable[[jax.Array], jax.Array]:
    layers = []
    for hidden_size in hidden_sizes:
        layers += [hk.Linear(hidden_size), activation]
    layers += [hk.Linear(output_size), output_activation]
    if squeeze_output:
        layers.append(partial(jnp.squeeze, axis=-1))
    return hk.Sequential(layers)


def mlp_with_bn(x: jax.Array, hidden_sizes: Sequence[int], output_size: int, activation: Activation, output_activation: Activation, *, squeeze_output: bool = False, 
                is_training: bool = True
        ) -> Callable[[jax.Array], jax.Array]:
    # BN = hk.BatchNorm
    BN = BatchRenorm
    # print(x.shape)
    # raise NotImplementedError
    x = BN(create_scale=True, create_offset=True, decay_rate=0.99, eps=1e-5)(x, is_training=is_training)
    for hidden_size in hidden_sizes:
        x = hk.Linear(hidden_size)(x)
        # print(x)
        x = activation(x)
        x = BN(create_scale=True, create_offset=True, decay_rate=0.99, eps=1e-5)(x, is_training=is_training)
    x = hk.Linear(output_size)(x)
    x = output_activation(x)
    if squeeze_output:
        x = jnp.squeeze(x, axis=-1)
    return x
def mlp_with_bn_q(x: jax.Array, hidden_sizes: Sequence[int], output_size: int, activation: Activation, output_activation: Activation, *, squeeze_output: bool = False, 
                is_training: bool = True
        ) -> Callable[[jax.Array], jax.Array]:
    # BN = hk.BatchNorm
    BN = BatchRenorm
    # print(x.shape)
    # raise NotImplementedError
    x = BN(create_scale=True, create_offset=True, decay_rate=0.99, eps=1e-5)(x, is_training=is_training)
    for hidden_size in hidden_sizes:
        x = hk.Linear(hidden_size)(x)
        # print(x)
        x = activation(x)
        x = BN(create_scale=True, create_offset=True, decay_rate=0.99, eps=1e-5)(x, is_training=is_training)
    x = hk.Linear(output_size)(x)
    x = output_activation(x)
    if squeeze_output:
        x = jnp.squeeze(x, axis=-1)
    return x

def mlp_with_bn_init(x: jax.Array, hidden_sizes: Sequence[int], output_size: int, activation: Activation, output_activation: Activation, *, squeeze_output: bool = False, 
                is_training: bool = True
        ) -> Callable[[jax.Array], jax.Array]:
    # BN = hk.BatchNorm
    init_W = hk.initializers.VarianceScaling(0.001)
    init_b = hk.initializers.RandomNormal(0.001)
    BN = BatchRenorm
    # print(x.shape)
    # raise NotImplementedError
    x = BN(create_scale=True, create_offset=True, decay_rate=0.99, eps=1e-3)(x, is_training=is_training)
    for hidden_size in hidden_sizes:
        x = hk.Linear(hidden_size, w_init=init_W, b_init=init_b)(x)
        # print(x)
        x = activation(x)
        x = BN(create_scale=True, create_offset=True, decay_rate=0.99, eps=1e-3)(x, is_training=is_training)
    x = hk.Linear(output_size, w_init=init_W, b_init=init_b)(x)
    x = output_activation(x)
    if squeeze_output:
        x = jnp.squeeze(x, axis=-1)
    return x
        


def scaled_sinusoidal_encoding(t: jax.Array, *, dim: int, theta: int = 10000, batch_shape = None) -> jax.Array:
    assert dim % 2 == 0
    if batch_shape is not None:
        assert is_broadcastable(jnp.shape(t), batch_shape)

    scale = 1 / dim ** 0.5
    half_dim = dim // 2
    freq_seq = jnp.arange(half_dim) / half_dim
    inv_freq = theta ** -freq_seq

    emb = jnp.einsum('..., j -> ... j', t, inv_freq)
    emb = jnp.concatenate((
        jnp.sin(emb),
        jnp.cos(emb),
    ), axis=-1)
    emb *= scale

    if batch_shape is not None:
        emb = jnp.broadcast_to(emb, (*batch_shape, dim))

    return emb
