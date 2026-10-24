"""Only injected Python fakes. Never loads a tensor library or model."""
import unittest

from experiments.paper_results_v1.native_adapters import (
    CallableTensorBackend, NativeAdapterError, RivaGANLoadedTensorBackend,
    RivaGANNativeAdapter, VideoSealLoadedBackend, VideoSealNativeAdapter,
)


class SealBackend:
    def __init__(self, output=None):
        self.output = output if output is not None else {"preds": [[[[0.1]], [[-0.2]], [[0.0]]]]}
        self.messages = []

    def build_message(self, bits):
        return [bits]

    def embed(self, media, message, output_uri=None):
        self.messages.append(message)
        return {"imgs_w": media}

    def embedded_media(self, value):
        return value['imgs_w']

    def extract(self, media):
        return self.output

    def media_identity(self, media):
        raise OSError('optional identity recording unavailable')


def seal(backend=None, **kwargs):
    return VideoSealNativeAdapter(backend or SealBackend(), native_message_length=2,
                                 backend_metadata={'detect_output_layout':'T,1+K,H,W'}, **kwargs)


class NativeInterfaces(unittest.TestCase):
    def test_native_length_and_bits_not_reconstructed(self):
        backend = SealBackend()
        adapter = seal(backend)
        _, record = adapter.run([0.1], [0,1])
        self.assertEqual(backend.messages, [[[0,1]]])
        self.assertIsNone(record['backend_metadata']['model_version'])
        self.assertEqual(record['embed']['embedded_media']['identity_status'], 'OPTIONAL_RECORD_UNAVAILABLE')
        self.assertEqual(record['extract']['raw_native_output'], backend.output)
        self.assertEqual(record['main_32bit_mapping'], 'PENDING_NOT_APPLIED')
        with self.assertRaises(NativeAdapterError):
            adapter.embed([0.1], [0])
        backend.build_message = lambda bits: [[1,0]]
        with self.assertRaises(NativeAdapterError):
            adapter.embed([0.1], [0,1])

    def test_sidecar_receives_full_value_and_rejects_nonfinite_first(self):
        received = []
        def store(label, value, descriptor):
            received.append(value)
            return {'lossless':True, 'uri':'saved.npz', 'format':'npz'}
        backend = SealBackend()
        record = seal(backend, inline_element_limit=1, native_output_store=store).extract(None)
        self.assertEqual(record['storage'], 'LOSSLESS_SIDECAR')
        self.assertIs(received[0], backend.output)
        self.assertEqual(record['total_element_count'], 3)
        backend.output['preds'][0][0][0][0] = float('nan')
        with self.assertRaises(NativeAdapterError):
            seal(backend, inline_element_limit=1, native_output_store=store).extract(None)
        self.assertEqual(len(received), 1)
        with self.assertRaises(NativeAdapterError):
            seal(inline_element_limit=1).extract(None)

    def test_output_shape_layout_and_numeric_type(self):
        for output in ({'preds':[[0.1,0.2]]}, {'preds':[[[[0.1]], [[False]], [[0.0]]]]}, {'missing':[]}):
            with self.assertRaises(NativeAdapterError):
                seal(SealBackend(output)).extract(None)
        with self.assertRaises(NativeAdapterError):
            VideoSealNativeAdapter(SealBackend(), native_message_length=2, backend_metadata={})

    def test_optional_metadata_nonfinite_is_record_only(self):
        adapter = VideoSealNativeAdapter(SealBackend(), native_message_length=2,
             backend_metadata={'detect_output_layout':'T,1+K,H,W', 'model_version':float('nan')})
        self.assertIsNone(adapter.backend_metadata['model_version'])
        self.assertIn('model_version', adapter.backend_metadata['optional_record_issues'])
        self.assertEqual(adapter.extract(None)['storage'], 'INLINE_FULL')

    def test_rivagan_soft_outputs_transport_and_native_tie(self):
        transport = {'transport_mode':'EXPLICIT_TENSOR','color_layout':'RGB_UINT8_TO_BGR',
                     'decoder_normalization':'uint8/127.5-1','codec_comparability':'NO_HIDDEN_CODEC'}
        submitted=[]
        def embed(media,bits,output_uri=None):
            submitted.append(bits)
            return media
        backend=CallableTensorBackend(embed,lambda media:[[-0.2,0.0,0.4,0.8]*8],transport_metadata=transport)
        adapter=RivaGANNativeAdapter(backend,backend_metadata={})
        _,record=adapter.run([0.1],[0,1]*16)
        self.assertEqual(submitted,[tuple([0,1]*16)])
        self.assertEqual(record['backend_metadata']['transport'],transport)
        self.assertEqual(record['extract']['frame_soft_outputs'][0][:4],[-0.2,0.0,0.4,0.8])
        self.assertEqual(record['extract']['frame_native_zero_threshold_bits'][0][:4],[0,1,1,1])
        self.assertEqual(record['sequence_32bit_recovery'],'PENDING_REDUCER_NOT_APPLIED')
        for output in ([[0.0]*31], [[float('inf')]*32], [[True]*32], []):
            backend.extract_call=lambda media, result=output:result
            with self.assertRaises(NativeAdapterError):
                adapter.extract(None)
        with self.assertRaises(NativeAdapterError):
            RivaGANNativeAdapter(CallableTensorBackend(embed,lambda media:[],transport_metadata={}),backend_metadata={})

    def test_injected_tensor_rgb_bgr_normalization_and_actual_validation(self):
        calls=[]
        class Frame:
            shape=(1,1,3)
            def __getitem__(self,index):
                calls.append(('reverse_rgb',index[-1].step))
                return self
            def copy(self):return self
        class Tensor:
            dtype='float32'
            shape=(1,32)
            def __truediv__(self,x):calls.append(('divide',x));return self
            def __sub__(self,x):calls.append(('subtract',x));return self
            def permute(self,*args):calls.append(('permute',args));return self
            def unsqueeze(self,x):calls.append(('unsqueeze',x));return self
        class Module:
            float32='float32'
            def as_tensor(self,value,**kwargs):return Tensor()
        backend=RivaGANLoadedTensorBackend(object(),tensor_module=Module(),array_module=object(),device='fake',input_color_layout='RGB_UINT8_TO_BGR')
        backend._model_frame(Frame())
        self.assertEqual(calls[:3],[('reverse_rgb',-1),('divide',127.5),('subtract',1.0)])
        class Frames:
            shape=(1,1,1,3)
            dtype='uint8'
            def __iter__(self):return iter([Frame()])
        self.assertIsInstance(backend._frames(Frames()),Frames)
        class WrongDtype(Frames):dtype='float32'
        with self.assertRaises(NativeAdapterError):backend._frames(WrongDtype())
        class BadEncoded:
            shape=(1,3,1,1,1)
            dtype='float32'
            def tolist(self):return [[[[[float('nan')]]],[[[0.0]]],[[[0.0]]]]]
        class Model:
            def encoder(self,*args):return BadEncoded()
        backend.model=Model()
        with self.assertRaises(NativeAdapterError):backend.embed(Frames(),[0]*32)

    def test_videoseal_loaded_tensor_shape_dtype_range(self):
        class Media:
            shape=(1,3,1,1)
            dtype='float32'
            def __init__(self,value):self.value=value
            def tolist(self):return [[[[self.value]], [[0.0]], [[1.0]]]]
        class Model:
            def detect(self,media,**kwargs):return {'preds':[[[[0.1]], [[0.2]], [[0.3]]]]}
        backend=VideoSealLoadedBackend(Model(),lambda bits:[bits])
        self.assertIn('preds',backend.extract(Media(0.5)))
        for value in (-0.1,1.1,float('nan')):
            with self.assertRaises(NativeAdapterError):backend.extract(Media(value))
        bad=Media(0.5);bad.dtype='uint8'
        with self.assertRaises(NativeAdapterError):backend.extract(bad)

    def test_videoseal_embed_validates_output_before_extract(self):
        class Media:
            shape=(1,3,1,1)
            dtype='float32'
            def __init__(self,value=0.5):self.value=value
            def tolist(self):return [[[[self.value]], [[0.0]], [[1.0]]]]
        class Model:
            output=Media()
            def embed(self,*args,**kwargs):return {'imgs_w':self.output}
            def detect(self,*args,**kwargs):raise AssertionError('embed validation must not call extract')
        model=Model()
        adapter=VideoSealNativeAdapter(VideoSealLoadedBackend(model,lambda bits:[bits]),
                 native_message_length=2,backend_metadata={'detect_output_layout':'T,1+K,H,W'})
        output,_=adapter.embed(Media(),[0,1])
        self.assertIs(output,model.output)
        for value in (float('nan'),float('inf')):
            model.output=Media(value)
            with self.assertRaises(NativeAdapterError):adapter.embed(Media(),[0,1])
        model.output=Media();model.output.dtype='uint8'
        with self.assertRaises(NativeAdapterError):adapter.embed(Media(),[0,1])
        model.output=Media();model.output.shape=(2,3,1,1)
        with self.assertRaises(NativeAdapterError):adapter.embed(Media(),[0,1])
        # Finite native values are not implicitly clipped by output validation.
        model.output=Media(1.2)
        output,_=adapter.embed(Media(),[0,1])
        self.assertEqual(output.value,1.2)


if __name__ == '__main__':
    unittest.main()
