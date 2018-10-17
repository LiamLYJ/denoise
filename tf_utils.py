import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import random
import math


# prepare for data loader
def select_raw(img_pure, select_ch):
    if select_ch is 'R':
        img = img_pure[0::2, 0::2]
    elif select_ch is 'RG':
        img = img_pure[0::2, 1::2]
    elif select_ch is 'GB':
        img = img_pure[1::2, 0::2]
    elif select_ch is 'B':
        img = img_pure[1::2, 1::2]
    else:
        raise ValueError('oops!, wrong select_ch: %s'%(select_ch))
    return img

def make_stack_hqjitter(image, height, width, depth, burst_length, to_shift, upscale, jitter):
    j_up = jitter * upscale
    h_up = height * upscale + 2 * j_up
    w_up = width * upscale + 2 * j_up
    v_error = tf.maximum((h_up - tf.shape(image)[0] + 1) // 2, 0)
    h_error = tf.maximum((w_up - tf.shape(image)[1] + 1) // 2, 0)
    image = tf.pad(image, [[v_error, v_error], [h_error, h_error], [0, 0]])

    stack = []
    for i in range(depth):
        stack.append(tf.random_crop(image, [h_up, w_up, 1]))
    stack = tf.stack(stack, axis=0)
    return stack


def make_batch_hqjitter(patches, burst_length, batch_size, repeats, height, width,
                        to_shift, upscale, jitter, smalljitter):
    # patches is [burst_length, h_up, w_up, 3]
    j_up = jitter * upscale
    h_up = height * upscale  # + 2 * j_up
    w_up = width * upscale  # + 2 * j_up

    bigj_patches = patches
    # print ('bigj_patches: ', bigj_patches.shape)
    delta_up = (jitter - smalljitter) * upscale
    smallj_patches = patches[:, delta_up:-delta_up, delta_up:-delta_up, ...]

    unique = batch_size//repeats
    batch = []
    for i in range(unique):
        for j in range(repeats):
            curr = [patches[i, j_up:-j_up, j_up:-j_up, :]]
            prob = tf.minimum(tf.cast(tf.random_poisson(
                1.5, []), tf.float32)/burst_length, 1.)
            for k in range(burst_length - 1):
                flip = tf.random_uniform([])
                p2use = tf.cond(flip < prob, lambda: bigj_patches,
                                lambda: smallj_patches)
                # curr.append(tf.random_crop(p2use[i, ...], [h_up, w_up, 3]))
                curr.append(tf.random_crop(p2use[i, ...], [h_up, w_up, 1]))
            curr = tf.stack(curr, axis=0)
            curr = tf.image.resize_images(
                curr, [height, width], method=tf.image.ResizeMethod.AREA)
            curr = tf.transpose(curr, [1, 2, 3, 0])
            batch.append(curr)
    batch = tf.stack(batch, axis=0)
    # print ('batch shape: ', batch.shape)
    return batch


# sRGB convertion
def sRGBforward(x):
    b = .0031308
    gamma = 1./2.4
    # a = .055
    # k0 = 12.92
    a = 1./(1./(b**gamma*(1.-gamma))-1.)
    k0 = (1+a)*gamma*b**(gamma-1.)

    def gammafn(x): return (1+a)*tf.pow(tf.maximum(x, b), gamma)-a
    # gammafn = lambda x : (1.-k0*b)/(1.-b)*(x-1.)+1.
    srgb = tf.where(x < b, k0*x, gammafn(x))
    k1 = (1+a)*gamma
    srgb = tf.where(x > 1, k1*x-k1+1, srgb)
    return srgb


# batch Downsample
def batch_down2(img):
    return (img[:, ::2, ::2, ...]+img[:, 1::2, ::2, ...]+img[:, ::2, 1::2, ...]+img[:, 1::2, 1::2, ...])/4


# Loss
def gradient(imgs):
    return tf.stack([.5*(imgs[..., 1:, :-1,:]-imgs[..., :-1, :-1,:]),
                     .5*(imgs[..., :-1, 1:,:]-imgs[..., :-1, :-1,:])], axis=-1)


def gradient_loss(guess, truth):
    return tf.reduce_mean(tf.abs(gradient(guess)-gradient(truth)))


def basic_img_loss(img, truth):
    l2_pixel = tf.reduce_mean(tf.square(img - truth))
    l1_grad = gradient_loss(img, truth)
    return l2_pixel + l1_grad

def filt_reg_loss(filts, final_K, initial_W, final_W, weight):
    weight = max(0.0, weight)
    fsh = tf.shape(filts)
    filts = tf.reshape(filts, [fsh[0], fsh[1], fsh[2], final_K ** 2 * initial_W, final_W])
    target = tf.ones([fsh[1], fsh[2], final_W])
    filts_sum = tf.reduce_sum(filts,axis = [3])
    loss = tf.reduce_mean(tf.abs(target - filts_sum))
    return loss * weight

def convolve(img_stack, filts, final_K, final_W):
    initial_W = img_stack.get_shape().as_list()[-1]
    imgsh = tf.shape(img_stack)
    fsh = tf.shape(filts)
    filts = tf.reshape(filts, [fsh[0],fsh[1],fsh[2],-1])
    img_stack = tf.cond(tf.less(fsh[1], imgsh[1]), lambda: batch_down2(img_stack), lambda: img_stack)
    # print ('filts shape: ', filts.shape)
    filts = tf.reshape(
        filts, [fsh[0], fsh[1], fsh[2], final_K ** 2 * initial_W, final_W])

    kpad = final_K//2
    imgs = tf.pad(img_stack, [[0, 0], [kpad, kpad], [kpad, kpad], [0, 0]])
    ish = tf.shape(img_stack)
    img_stack = []
    for i in range(final_K):
        for j in range(final_K):
            img_stack.append(
                imgs[:, i:tf.shape(imgs)[1]-2*kpad+i, j:tf.shape(imgs)[2]-2*kpad+j, :])
    img_stack = tf.stack(img_stack, axis=-2)
    img_stack = tf.reshape(
        img_stack, [ish[0], ish[1], ish[2], final_K**2 * initial_W, 1])
    # removes the final_K**2*initial_W dimension but keeps final_W
    img_net = tf.reduce_sum(img_stack * filts, axis=-2)
    return img_net


def sep_convolve(img, filts, final_K, final_W):
    pre_img = img * filts
    return pre_img

def convolve_per_layer(input_stack, filts, final_K, final_W):
    initial_W = input_stack.get_shape().as_list()[-1]
    filts = tf.reshape(filts, [tf.shape(input_stack)[0], tf.shape(input_stack)[1], tf.shape(input_stack)[
                   2], final_K, final_K, initial_W, final_W])
    img_net = []
    for i in range(initial_W):
        img_net.append(
            convolve(input_stack[..., i:i+1], filts[..., i:i+1, :], final_K, final_W))
    img_net = tf.concat(img_net, axis=-1)
    return img_net

def data_augment(images,
            horizontal_flip=False,
            vertical_flip=False,
            rotate=0, # Maximum rotation angle in degrees
            crop_probability=0, # How often we do crops
            crop_min_percent=0.6, # Minimum linear dimension of a crop
            crop_max_percent=1.,  # Maximum linear dimension of a crop
            mixup=0):  # Mixup coeffecient, see https://arxiv.org/abs/1710.09412.pdf

  # if images.dtype != tf.float32:
  #   images = tf.image.convert_image_dtype(images, dtype=tf.float32)
  #   images = tf.subtract(images, 0.5)
  #   images = tf.multiply(images, 2.0)

  with tf.name_scope('augmentation'):
    shp = tf.shape(images)
    batch_size, height, width = shp[0], shp[1], shp[2]
    width = tf.cast(width, tf.float32)
    height = tf.cast(height, tf.float32)

    # The list of affine transformations that our image will go under.
    # Every element is Nx8 tensor, where N is a batch size.
    transforms = []
    identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
    if horizontal_flip:
      coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
      flip_transform = tf.convert_to_tensor(
          [-1., 0., width, 0., 1., 0., 0., 0.], dtype=tf.float32)
      transforms.append(
          tf.where(coin,
                   tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                   tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

    if vertical_flip:
      coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
      flip_transform = tf.convert_to_tensor(
          [1, 0, 0, 0, -1, height, 0, 0], dtype=tf.float32)
      transforms.append(
          tf.where(coin,
                   tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                   tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

    if rotate > 0:
      angle_rad = rotate / 180 * math.pi
      angles = tf.random_uniform([batch_size], -angle_rad, angle_rad)
      transforms.append(
          tf.contrib.image.angles_to_projective_transforms(
              angles, height, width))

    if crop_probability > 0:
      crop_pct = tf.random_uniform([batch_size], crop_min_percent,
                                   crop_max_percent)
      left = tf.random_uniform([batch_size], 0, width * (1 - crop_pct))
      top = tf.random_uniform([batch_size], 0, height * (1 - crop_pct))
      crop_transform = tf.stack([
          crop_pct,
          tf.zeros([batch_size]), top,
          tf.zeros([batch_size]), crop_pct, left,
          tf.zeros([batch_size]),
          tf.zeros([batch_size])
      ], 1)

      coin = tf.less(
          tf.random_uniform([batch_size], 0, 1.0), crop_probability)
      transforms.append(
          tf.where(coin, crop_transform,
                   tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

    if transforms:
      images = tf.contrib.image.transform(
          images,
          tf.contrib.image.compose_transforms(*transforms),
          interpolation='BILINEAR') # or 'NEAREST'

    def cshift(values): # Circular shift in batch dimension
      return tf.concat([values[-1:, ...], values[:-1, ...]], 0)

    if mixup > 0:
      mixup = 1.0 * mixup # Convert to float, as tf.distributions.Beta requires floats.
      beta = tf.distributions.Beta(mixup, mixup)
      lam = beta.sample(batch_size)
      ll = tf.expand_dims(tf.expand_dims(tf.expand_dims(lam, -1), -1), -1)
      images = ll * images + (1 - ll) * cshift(images)

  return images


# For separable stuff
def convolve_aniso(img_stack, filts, final_Kh, final_Kw, final_W, layerwise=False):
    initial_W = img_stack.get_shape().as_list()[-1]

    fsh = tf.shape(filts)
    if layerwise:
        filts = tf.reshape(
            filts, [fsh[0], fsh[1], fsh[2], final_Kh * final_Kw, initial_W])
    else:
        filts = tf.reshape(
            filts, [fsh[0], fsh[1], fsh[2], final_Kh * final_Kw * initial_W, final_W])

    kpadh = final_Kh//2
    kpadw = final_Kw//2
    imgs = tf.pad(img_stack, [[0, 0], [kpadh, kpadh], [kpadw, kpadw], [0, 0]])
    ish = tf.shape(img_stack)
    img_stack = []
    for i in range(final_Kh):
        for j in range(final_Kw):
            img_stack.append(
                imgs[:, i:tf.shape(imgs)[1]-2*kpadh+i, j:tf.shape(imgs)[2]-2*kpadw+j, :])
    img_stack = tf.stack(img_stack, axis=-2)
    if layerwise:
        img_stack = tf.reshape(
            img_stack, [ish[0], ish[1], ish[2], final_Kh * final_Kw, initial_W])
    else:
        img_stack = tf.reshape(
            img_stack, [ish[0], ish[1], ish[2], final_Kh * final_Kw * initial_W, 1])
    # removes the final_K**2*initial_W dimension but keeps final_W
    img_net = tf.reduce_sum(img_stack * filts, axis=-2)
    return img_net



# SSIM


def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 +
                              1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma)  # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1, 1, 1, 1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1, 1, 1, 1], padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(
        img1*img1, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(
        img2*img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(
        img1*img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1) *
                                                      (sigma1_sq + sigma2_sq + C2)),
                 (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1) *
                                                     (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


# summary helpers
def filts2imgs(filts, h, w):
    K = tf.shape(filts)[1]
    ch = tf.shape(filts)[3]
    filts = tf.reshape(filts, [-1, K, K, h, w])
    filts = tf.pad(filts, [[0, 0], [1, 1], [1, 1], [0, 0], [0, 0]])
    filts = tf.transpose(filts, [0, 3, 1, 4, 2])
    filts = tf.reshape(filts, [-1, h*(K+2), w*(K+2), 1])
    return filts


def store_plot(plots, name, scalar, label=""):
    if name not in plots:
        plots[name] = []
    plots[name].append([label, scalar])

    return plots


def gen_plots(plots, g_index):
    summaries = []
    for name in plots:
        plot = plots[name]
        # plot.sort(key=lambda x : x[0])
        scalars = []
        i = 0
        for label, scalar in plot:
            scalars.append(scalar)
            name += '_' + str(i) + '_' + label
            i += 1
            tensor = tf.reshape(tf.stack(scalars), [len(scalars)])
        scalar = tf.cond(g_index < len(scalars),
                         lambda: tensor[g_index], lambda: tensor[0])
        summaries.append(tf.summary.scalar(name, scalar))
        print (('Generating plot with name', name))
    return tf.summary.merge(summaries)

def run_summaries(sess, writer, summaries, step, fdict = None):
    if fdict is None:
        summaries_out, = sess.run([summaries])
    else:
        summaries_out, = sess.run([summaries], feed_dict = fdict)
    writer.add_summary(summaries_out, step)

# HDR_PLUS
def rcwindow(N):
    x = tf.linspace(0., N, N+1)[:-1]
    rcw = .5 - .5 * tf.cos(2.*np.pi * (x + .5) / N)
    rcw = tf.reshape(rcw, (N, 1)) * tf.reshape(rcw, (1, N))
    return rcw


def roll_tf(x, shift, axis=0):
    sh = tf.shape(x)
    n = sh[axis]
    shift = shift % n
    bl0 = tf.concat([sh[:axis], [n-shift], sh[axis+1:]], axis=0)
    bl1 = tf.concat([sh[:axis], [shift],   sh[axis+1:]], axis=0)
    or0 = tf.concat([tf.zeros_like(sh[:axis]), [shift],
                     tf.zeros_like(sh[axis+1:])], axis=0)
    or1 = tf.zeros_like(bl0)
    x0 = tf.slice(x, or0, bl0)
    x1 = tf.slice(x, or1, bl1)
    return tf.concat([x0, x1], axis=axis)


def hdrplus_merge(imgs, N, c, sig):
    def ccast_tf(x): return tf.complex(x, tf.zeros_like(x))

    # imgs is [batch, h, w, ch]
    rcw = tf.expand_dims(rcwindow(N), axis=-1)
    imgs = imgs * rcw
    imgs = tf.transpose(imgs, [0, 3, 1, 2])
    imgs_f = tf.fft2d(ccast_tf(imgs))
    imgs_f = tf.transpose(imgs_f, [0, 2, 3, 1])
    Dz2 = tf.square(tf.abs(imgs_f[..., 0:1] - imgs_f))
    Az = Dz2 / (Dz2 + c*sig**2)
    filt0 = 1 + tf.expand_dims(tf.reduce_sum(Az[..., 1:], axis=-1), axis=-1)
    filts = tf.concat([filt0, 1 - Az[..., 1:]], axis=-1)
    output_f = tf.reduce_mean(imgs_f * ccast_tf(filts), axis=-1)
    output_f = tf.real(tf.ifft2d(output_f))

    return output_f


def hdrplus_tiled(noisy, N, sig, c=10**2.25):
    sh = tf.shape(noisy)[0:3]
    buffer = tf.zeros_like(noisy[..., 0])
    allpics = []
    for i in range(2):
        for j in range(2):
            nrolled = roll_tf(roll_tf(noisy, shift=-N//2*i,
                                      axis=1), shift=-N//2*j, axis=2)
            hpatches = (tf.transpose(tf.reshape(
                nrolled, [sh[0], sh[1]//N, N, sh[2]//N, N, -1]), [0, 1, 3, 2, 4, 5]))
            hpatches = tf.reshape(
                hpatches, [sh[0]*sh[1]*sh[2]//N**2, N, N, -1])
            merged = hdrplus_merge(hpatches, N, c, sig)
            merged = tf.reshape(merged, [sh[0], sh[1]//N, sh[2]//N, N, N])
            merged = (tf.reshape(tf.transpose(merged, [0, 1, 3, 2, 4]), sh))
            merged = roll_tf(roll_tf(merged, axis=1, shift=N //
                                     2*i), axis=2, shift=N//2*j)
            buffer += merged
            allpics.append(merged)
    return buffer
