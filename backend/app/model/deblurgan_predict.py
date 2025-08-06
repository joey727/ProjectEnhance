import os
from PIL import Image
import numpy as np
import tensorflow.compat.v1 as tf  # type: ignore
tf.disable_v2_behavior()


class DeblurGANPredictor:
    def __init__(self, checkpoint_dir):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            # Check for available checkpoint files
            checkpoint_files = [f for f in os.listdir(
                checkpoint_dir) if f.endswith('.meta')]
            if not checkpoint_files:
                raise FileNotFoundError(
                    f"No .meta files found in {checkpoint_dir}")

            # Use the first available .meta file (or you can specify which one to use)
            meta_file = checkpoint_files[0]
            ckpt_name = meta_file.replace('.meta', '')
            ckpt_path = os.path.join(checkpoint_dir, ckpt_name)

            print(f"Loading checkpoint: {ckpt_path}")

            # Import meta graph with clear devices to avoid GPU/CPU conflicts
            saver = tf.train.import_meta_graph(
                os.path.join(checkpoint_dir, meta_file),
                clear_devices=True
            )
            saver.restore(self.sess, ckpt_path)

            # Print all operations to debug
            print("Available operations in graph:")
            for op in self.graph.get_operations():
                # Skip gradient operations
                if not op.name.startswith('gradients/'):
                    print(f"  {op.name}")

            # Try to find the correct input tensor name
            try:
                self.input_image = self.graph.get_tensor_by_name(
                    'input_image:0')
            except KeyError:
                # If 'input_image:0' doesn't exist, try to find the input tensor
                input_tensors = []
                for op in self.graph.get_operations():
                    # Look for placeholder operations that are not gradients
                    if (op.type == 'Placeholder'
                        and not op.name.startswith('gradients/')
                        and not 'grad' in op.name.lower()
                            and op.outputs):
                        input_tensors.append(op.outputs[0].name)

                if not input_tensors:
                    # Fallback: look for any operation that might be an input
                    for op in self.graph.get_operations():
                        if (('input' in op.name.lower() or 'placeholder' in op.name.lower())
                            and not op.name.startswith('gradients/')
                            and not 'grad' in op.name.lower()
                                and op.outputs):
                            input_tensors.append(op.outputs[0].name)

                if input_tensors:
                    print(f"Available input tensors: {input_tensors}")
                    # Choose the first one that looks like a proper input
                    for tensor_name in input_tensors:
                        try:
                            tensor = self.graph.get_tensor_by_name(tensor_name)
                            # Check if it has a reasonable shape for input
                            if tensor.shape.as_list() and len(tensor.shape.as_list()) == 4:
                                self.input_image = tensor
                                print(f"Selected input tensor: {tensor_name}")
                                break
                        except:
                            continue
                    else:
                        raise ValueError(
                            "Could not find suitable input tensor in the graph")
                else:
                    raise ValueError(
                        "Could not find input tensor in the graph")

            try:
                self.output_image = self.graph.get_tensor_by_name(
                    'generator/output_image:0')
            except KeyError:
                # Try to find the correct output tensor
                # Look for the final output tensor in the generator
                output_candidates = [
                    'generator/clip_by_value:0',  # Final clipped output
                    'generator/add_32:0',         # Final addition
                    'generator/Tanh:0',           # Tanh activation
                    'generator/conv_last/add:0'   # Last conv layer output
                ]

                for candidate in output_candidates:
                    try:
                        tensor = self.graph.get_tensor_by_name(candidate)
                        if tensor.shape.as_list() and len(tensor.shape.as_list()) == 4:
                            self.output_image = tensor
                            print(f"Selected output tensor: {candidate}")
                            break
                    except:
                        continue
                else:
                    # Fallback: look for any operation that might be an output
                    output_tensors = []
                    for op in self.graph.get_operations():
                        if (('output' in op.name.lower() or 'generator' in op.name.lower())
                            and not op.name.startswith('gradients/')
                            and not 'grad' in op.name.lower()
                            and not 'mirror' in op.name.lower()  # Skip padding operations
                                and op.outputs):
                            output_tensors.append(op.outputs[0].name)

                    if output_tensors:
                        print(f"Available output tensors: {output_tensors}")
                        # Choose the first one that looks like a proper output
                        for tensor_name in output_tensors:
                            try:
                                tensor = self.graph.get_tensor_by_name(
                                    tensor_name)
                                # Check if it has a reasonable shape for output
                                if tensor.shape.as_list() and len(tensor.shape.as_list()) == 4:
                                    self.output_image = tensor
                                    print(
                                        f"Selected output tensor: {tensor_name}")
                                    break
                            except:
                                continue
                        else:
                            raise ValueError(
                                "Could not find suitable output tensor in the graph")
                    else:
                        raise ValueError(
                            "Could not find output tensor in the graph")

    def predict(self, pil_image):
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # For DeblurGAN, we need to simulate a blurred input
        # The model expects a blurred image as input and outputs a sharp image
        # Since we're getting a regular image, we'll apply a slight blur to simulate the input

        # Apply a slight Gaussian blur to simulate the "blurred" input
        from PIL import ImageFilter
        blurred_input = pil_image.filter(ImageFilter.GaussianBlur(radius=1.5))

        # Resize to 256x256
        blurred_input = blurred_input.resize((256, 256))

        # Convert to numpy and normalize to [-1, 1] range (typical for GANs)
        img = np.array(blurred_input).astype(np.float32)
        img = (img / 127.5) - 1.0  # Normalize to [-1, 1]

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        print(f"Input image shape: {img.shape}")
        print(f"Input image range: [{img.min():.3f}, {img.max():.3f}]")

        with self.graph.as_default():
            try:
                # Run inference
                output = self.sess.run(self.output_image, feed_dict={
                    self.input_image: img
                })

                print(f"Output shape: {output.shape}")
                print(
                    f"Output range: [{output.min():.3f}, {output.max():.3f}]")

                # Handle different output formats
                if len(output.shape) == 4:
                    output_img = output[0]  # Remove batch dimension
                else:
                    output_img = output

                # Convert from [-1, 1] range back to [0, 255]
                output_img = ((output_img + 1) * 127.5).astype(np.uint8)
                output_img = np.clip(output_img, 0, 255)

                print(
                    f"Final output range: [{output_img.min()}, {output_img.max()}]")

                # Convert back to PIL Image
                result_image = Image.fromarray(output_img)

                return result_image

            except Exception as e:
                print(f"Error during prediction: {e}")
                # Return the original image as fallback
                return pil_image
