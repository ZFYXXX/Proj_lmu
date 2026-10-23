"""Exact video evidence provenance; historical details live in referenced JSON.

Each entry is explicitly registered. Adding an algorithm or backbone elsewhere
cannot create passing evidence. Keep record kinds, revisions and dates exact.
This index is ordinary package code and needs no runtime artifact-file access.
"""

from types import MappingProxyType
from mllm_comp.core.support import EvidenceKind, EvidenceRecord


# key -> (artifact checksum, (kind, source revision, recorded time) records)
VIDEO_EVIDENCE = MappingProxyType({
    ('metok', 'llava_onevision', 'forward'): (
        '82e99d5ec9f4f7d6067e8eb56dc0700fefa0e26c8314df91835503c2e029b7be',
        (
            (EvidenceKind.KERNEL, '1dad51378a5e2a2cbdf5b962cc2a1a430811cfdc', '2026-08-27T18:14:53+08:00'),
            (EvidenceKind.CPU_CONTRACT, '1dad51378a5e2a2cbdf5b962cc2a1a430811cfdc', '2026-08-27T18:14:53+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, '1dad51378a5e2a2cbdf5b962cc2a1a430811cfdc', '2026-08-27T18:14:53+08:00'),
            (EvidenceKind.TRAIN_STEP, '1dad51378a5e2a2cbdf5b962cc2a1a430811cfdc', '2026-08-27T18:14:53+08:00'),
            (EvidenceKind.TRAIN_BACKWARD, '1dad51378a5e2a2cbdf5b962cc2a1a430811cfdc', '2026-08-27T18:14:53+08:00'),
            (EvidenceKind.CHECKPOINT_RELOAD, '1dad51378a5e2a2cbdf5b962cc2a1a430811cfdc', '2026-08-27T18:14:53+08:00'),
        ),
    ),
    ('metok', 'llava_onevision', 'generate'): (
        '9f1468af846d8a47ceb3e5dfe45978c324e9533b98a1ec07f63a2407ffba07e9',
        (
            (EvidenceKind.KERNEL, '245dc7e07121e9b125d35928cf9d33fa8729a404', '2026-08-26T15:11:52+08:00'),
            (EvidenceKind.CPU_CONTRACT, '245dc7e07121e9b125d35928cf9d33fa8729a404', '2026-08-26T15:11:52+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, '245dc7e07121e9b125d35928cf9d33fa8729a404', '2026-08-26T15:11:52+08:00'),
        ),
    ),
    ('metok', 'qwen25_vl', 'forward'): (
        '8bc9d7f9208a8cbeb362e25cc241f4b2566739389f39a4be4c766b609017b2de',
        (
            (EvidenceKind.KERNEL, '76022bd15793b815da5c6b5e88687fd1747012dc', '2026-08-27T22:22:08+08:00'),
            (EvidenceKind.CPU_CONTRACT, '76022bd15793b815da5c6b5e88687fd1747012dc', '2026-08-27T22:22:08+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, '76022bd15793b815da5c6b5e88687fd1747012dc', '2026-08-27T22:22:08+08:00'),
            (EvidenceKind.TRAIN_STEP, '76022bd15793b815da5c6b5e88687fd1747012dc', '2026-08-27T22:22:08+08:00'),
            (EvidenceKind.TRAIN_BACKWARD, '76022bd15793b815da5c6b5e88687fd1747012dc', '2026-08-27T22:22:08+08:00'),
            (EvidenceKind.CHECKPOINT_RELOAD, '76022bd15793b815da5c6b5e88687fd1747012dc', '2026-08-27T22:22:08+08:00'),
        ),
    ),
    ('metok', 'qwen25_vl', 'generate'): (
        '0ba61a065a3e3f93c5357efb6a01734300c3eecd0264f730b686b50eb16c4122',
        (
            (EvidenceKind.KERNEL, '2076b57571c4ca3ed78a4c8765a58381e34281be', '2026-08-26T16:16:23+08:00'),
            (EvidenceKind.CPU_CONTRACT, '2076b57571c4ca3ed78a4c8765a58381e34281be', '2026-08-26T16:16:23+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, '2076b57571c4ca3ed78a4c8765a58381e34281be', '2026-08-26T16:16:23+08:00'),
        ),
    ),
    ('metok', 'qwen3_vl', 'forward'): (
        '5d631e56e26ec37e6ad970ef00c5de0bb17c2cf29bef69af363b1e617930bd7f',
        (
            (EvidenceKind.KERNEL, 'da320e459f5225e101f56735fd18b91da0307159', '2026-08-27T19:19:40+08:00'),
            (EvidenceKind.CPU_CONTRACT, 'da320e459f5225e101f56735fd18b91da0307159', '2026-08-27T19:19:40+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, 'da320e459f5225e101f56735fd18b91da0307159', '2026-08-27T19:19:40+08:00'),
            (EvidenceKind.TRAIN_STEP, 'da320e459f5225e101f56735fd18b91da0307159', '2026-08-27T19:19:40+08:00'),
            (EvidenceKind.TRAIN_BACKWARD, 'da320e459f5225e101f56735fd18b91da0307159', '2026-08-27T19:19:40+08:00'),
            (EvidenceKind.CHECKPOINT_RELOAD, 'da320e459f5225e101f56735fd18b91da0307159', '2026-08-27T19:19:40+08:00'),
        ),
    ),
    ('metok', 'qwen3_vl', 'generate'): (
        '7fd3846660a803a58a97e23050e00dcc733ab68b748ef23e2e0fd4f2a49bb5da',
        (
            (EvidenceKind.KERNEL, 'f2dc5b21116eb6d0ecb57e015f6f8b027884f6b2', '2026-08-26T17:12:17+08:00'),
            (EvidenceKind.CPU_CONTRACT, 'f2dc5b21116eb6d0ecb57e015f6f8b027884f6b2', '2026-08-26T17:12:17+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, 'f2dc5b21116eb6d0ecb57e015f6f8b027884f6b2', '2026-08-26T17:12:17+08:00'),
        ),
    ),
    ('prunevid', 'llava_onevision', 'forward'): (
        '5605ad42c389ccca7379463c8b69022d534a4b94febba3dcf079f42aca5ce406',
        (
            (EvidenceKind.KERNEL, 'b4b11e392edafafe0baad55df901b369affeec9d', '2026-08-27T15:41:19+08:00'),
            (EvidenceKind.CPU_CONTRACT, 'b4b11e392edafafe0baad55df901b369affeec9d', '2026-08-27T15:41:19+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, 'b4b11e392edafafe0baad55df901b369affeec9d', '2026-08-27T15:41:19+08:00'),
            (EvidenceKind.TRAIN_STEP, 'b4b11e392edafafe0baad55df901b369affeec9d', '2026-08-27T15:41:19+08:00'),
            (EvidenceKind.TRAIN_BACKWARD, 'b4b11e392edafafe0baad55df901b369affeec9d', '2026-08-27T15:41:19+08:00'),
            (EvidenceKind.CHECKPOINT_RELOAD, 'b4b11e392edafafe0baad55df901b369affeec9d', '2026-08-27T15:41:19+08:00'),
        ),
    ),
    ('prunevid', 'llava_onevision', 'generate'): (
        'a6b215590c43054139a26e69c23d2b9c1dd177c8c06418dc83acfe952f5508b7',
        (
            (EvidenceKind.KERNEL, '3bb24b3bae744a667b57a7195dfd890f0f4a6bf7', '2026-08-26T12:14:13+08:00'),
            (EvidenceKind.CPU_CONTRACT, '3bb24b3bae744a667b57a7195dfd890f0f4a6bf7', '2026-08-26T12:14:13+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, '3bb24b3bae744a667b57a7195dfd890f0f4a6bf7', '2026-08-26T12:14:13+08:00'),
        ),
    ),
    ('prunevid', 'qwen25_vl', 'forward'): (
        'd27c7e0caed14a6f8665c4565006280fbc5d42deae43fc0b79f74c374fd4965e',
        (
            (EvidenceKind.KERNEL, '530575798f1b61ff45c02b5e6be3c157819ad183', '2026-08-27T16:41:33+08:00'),
            (EvidenceKind.CPU_CONTRACT, '530575798f1b61ff45c02b5e6be3c157819ad183', '2026-08-27T16:41:33+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, '530575798f1b61ff45c02b5e6be3c157819ad183', '2026-08-27T16:41:33+08:00'),
            (EvidenceKind.TRAIN_STEP, '530575798f1b61ff45c02b5e6be3c157819ad183', '2026-08-27T16:41:33+08:00'),
            (EvidenceKind.TRAIN_BACKWARD, '530575798f1b61ff45c02b5e6be3c157819ad183', '2026-08-27T16:41:33+08:00'),
            (EvidenceKind.CHECKPOINT_RELOAD, '530575798f1b61ff45c02b5e6be3c157819ad183', '2026-08-27T16:41:33+08:00'),
        ),
    ),
    ('prunevid', 'qwen25_vl', 'generate'): (
        '321f4b3dfd15e9c80445d542a1e32eb983bc6cca0feb9bc4080bf4b851ffd4b9',
        (
            (EvidenceKind.KERNEL, 'c4df7b4664e03ac254f5074a0a140e287d5b7d28', '2026-08-26T13:11:22+08:00'),
            (EvidenceKind.CPU_CONTRACT, 'c4df7b4664e03ac254f5074a0a140e287d5b7d28', '2026-08-26T13:11:22+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, 'c4df7b4664e03ac254f5074a0a140e287d5b7d28', '2026-08-26T13:11:22+08:00'),
        ),
    ),
    ('prunevid', 'qwen3_vl', 'forward'): (
        '0738565df99ae76f52256f81b53ffce19eebc942b82036edc2d603170b0e14ed',
        (
            (EvidenceKind.KERNEL, 'a1ae4079db0f4ed0b0b70583b08161e914736127', '2026-08-27T17:13:34+08:00'),
            (EvidenceKind.CPU_CONTRACT, 'a1ae4079db0f4ed0b0b70583b08161e914736127', '2026-08-27T17:13:34+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, 'a1ae4079db0f4ed0b0b70583b08161e914736127', '2026-08-27T17:13:34+08:00'),
            (EvidenceKind.TRAIN_STEP, 'a1ae4079db0f4ed0b0b70583b08161e914736127', '2026-08-27T17:13:34+08:00'),
            (EvidenceKind.TRAIN_BACKWARD, 'a1ae4079db0f4ed0b0b70583b08161e914736127', '2026-08-27T17:13:34+08:00'),
            (EvidenceKind.CHECKPOINT_RELOAD, 'a1ae4079db0f4ed0b0b70583b08161e914736127', '2026-08-27T17:13:34+08:00'),
        ),
    ),
    ('prunevid', 'qwen3_vl', 'generate'): (
        'bc606380501e41b311a1f898fa215eb67902769102257092be4db71b7fb05dcd',
        (
            (EvidenceKind.KERNEL, 'd096121325abce851ab4ddedbb8b4e5a32662287', '2026-08-26T14:02:07+08:00'),
            (EvidenceKind.CPU_CONTRACT, 'd096121325abce851ab4ddedbb8b4e5a32662287', '2026-08-26T14:02:07+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, 'd096121325abce851ab4ddedbb8b4e5a32662287', '2026-08-26T14:02:07+08:00'),
        ),
    ),
    ('visionzip', 'llava_onevision', 'forward'): (
        '869ceab9ed975cb15a162d04fc5f2fecf56adb6df40fcd8227325a8ad810bcda',
        (
            (EvidenceKind.KERNEL, 'e81d027e2ffbfd143cf27ad403f9c51c678081d6', '2026-08-27T12:20:27+08:00'),
            (EvidenceKind.CPU_CONTRACT, 'e81d027e2ffbfd143cf27ad403f9c51c678081d6', '2026-08-27T12:20:27+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, 'a3fc87d30e0a35d4bb4f7c227375ef5759d5fdd4', '2026-08-27T12:20:27+08:00'),
            (EvidenceKind.TRAIN_STEP, 'a3fc87d30e0a35d4bb4f7c227375ef5759d5fdd4', '2026-08-27T12:20:27+08:00'),
            (EvidenceKind.TRAIN_BACKWARD, 'a3fc87d30e0a35d4bb4f7c227375ef5759d5fdd4', '2026-08-27T12:20:27+08:00'),
            (EvidenceKind.CHECKPOINT_RELOAD, 'a3fc87d30e0a35d4bb4f7c227375ef5759d5fdd4', '2026-08-27T12:20:27+08:00'),
        ),
    ),
    ('visionzip', 'llava_onevision', 'generate'): (
        'ebfb6d8b905f1de9a9c30593ad0173ab741596b6e911d4bd1e2f008894d3c21d',
        (
            (EvidenceKind.KERNEL, 'e81d027e2ffbfd143cf27ad403f9c51c678081d6', '2026-08-26T08:02:07+08:00'),
            (EvidenceKind.CPU_CONTRACT, 'e81d027e2ffbfd143cf27ad403f9c51c678081d6', '2026-08-26T08:02:07+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, 'e81d027e2ffbfd143cf27ad403f9c51c678081d6', '2026-08-26T08:02:07+08:00'),
        ),
    ),
    ('visionzip', 'qwen25_vl', 'forward'): (
        '261a82a8ac417ca92469232575553594420c6a966dc2e21c53d3f0d30c0fea23',
        (
            (EvidenceKind.KERNEL, '6682c1f4f0ad02a292336ce72ab202c2e8d45756', '2026-08-27T13:13:15+08:00'),
            (EvidenceKind.CPU_CONTRACT, '6682c1f4f0ad02a292336ce72ab202c2e8d45756', '2026-08-27T13:13:15+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, 'a3cd5e2ea711c703a100261a66815d2edfd05a30', '2026-08-27T13:13:15+08:00'),
            (EvidenceKind.TRAIN_STEP, 'a3cd5e2ea711c703a100261a66815d2edfd05a30', '2026-08-27T13:13:15+08:00'),
            (EvidenceKind.TRAIN_BACKWARD, 'a3cd5e2ea711c703a100261a66815d2edfd05a30', '2026-08-27T13:13:15+08:00'),
            (EvidenceKind.CHECKPOINT_RELOAD, 'a3cd5e2ea711c703a100261a66815d2edfd05a30', '2026-08-27T13:13:15+08:00'),
        ),
    ),
    ('visionzip', 'qwen25_vl', 'generate'): (
        '463662ab0d46099bc02e21e4f5ddc39788c3921f08a7352cf2f90158ef3ec8b3',
        (
            (EvidenceKind.KERNEL, '6682c1f4f0ad02a292336ce72ab202c2e8d45756', '2026-08-26T08:38:16+08:00'),
            (EvidenceKind.CPU_CONTRACT, '6682c1f4f0ad02a292336ce72ab202c2e8d45756', '2026-08-26T08:38:16+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, '6682c1f4f0ad02a292336ce72ab202c2e8d45756', '2026-08-26T08:38:16+08:00'),
        ),
    ),
    ('visionzip', 'qwen3_vl', 'forward'): (
        '7aad9fbc105b6d9408c64ab58d93e88941e91c443fbf15a9ca93f92928065fae',
        (
            (EvidenceKind.KERNEL, '295f71d730696cd99a04252932119c7cade5b2b9', '2026-08-27T13:40:54+08:00'),
            (EvidenceKind.CPU_CONTRACT, '295f71d730696cd99a04252932119c7cade5b2b9', '2026-08-27T13:40:54+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, '67c1b870be5d41310a7a88f8a4a0f5e13affa0de', '2026-08-27T13:40:54+08:00'),
            (EvidenceKind.TRAIN_STEP, '67c1b870be5d41310a7a88f8a4a0f5e13affa0de', '2026-08-27T13:40:54+08:00'),
            (EvidenceKind.TRAIN_BACKWARD, '67c1b870be5d41310a7a88f8a4a0f5e13affa0de', '2026-08-27T13:40:54+08:00'),
            (EvidenceKind.CHECKPOINT_RELOAD, '67c1b870be5d41310a7a88f8a4a0f5e13affa0de', '2026-08-27T13:40:54+08:00'),
        ),
    ),
    ('visionzip', 'qwen3_vl', 'generate'): (
        '999dfff20f03674f545dc6b079e9979e20237c8fb47f1b0e11efdb6ffff45760',
        (
            (EvidenceKind.KERNEL, '295f71d730696cd99a04252932119c7cade5b2b9', '2026-08-26T09:24:05+08:00'),
            (EvidenceKind.CPU_CONTRACT, '295f71d730696cd99a04252932119c7cade5b2b9', '2026-08-26T09:24:05+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, '295f71d730696cd99a04252932119c7cade5b2b9', '2026-08-26T09:24:05+08:00'),
        ),
    ),
    ('fastvid', 'llava_onevision', 'forward'): (
        'ca5aec1f8c1ce495b20716923ba1f49c2e2db424a92d48be2413888f19f45662',
        (
            (EvidenceKind.KERNEL, 'e8cc5338801fcecdbe9d5831a5e274abe5539129', '2026-08-27T14:02:50+08:00'),
            (EvidenceKind.CPU_CONTRACT, 'e8cc5338801fcecdbe9d5831a5e274abe5539129', '2026-08-27T14:02:50+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, 'e8cc5338801fcecdbe9d5831a5e274abe5539129', '2026-08-27T14:02:50+08:00'),
            (EvidenceKind.TRAIN_STEP, 'e8cc5338801fcecdbe9d5831a5e274abe5539129', '2026-08-27T14:02:50+08:00'),
            (EvidenceKind.TRAIN_BACKWARD, 'e8cc5338801fcecdbe9d5831a5e274abe5539129', '2026-08-27T14:02:50+08:00'),
            (EvidenceKind.CHECKPOINT_RELOAD, 'e8cc5338801fcecdbe9d5831a5e274abe5539129', '2026-08-27T14:02:50+08:00'),
        ),
    ),
    ('fastvid', 'llava_onevision', 'generate'): (
        '94cc24d0e7ae8d188c809575f7f214f6ff11c4517996938b6780814b26db3887',
        (
            (EvidenceKind.KERNEL, '3b47797fdfc594bb51a349137a0e9faa59ede1fa', '2026-08-26T10:02:52+08:00'),
            (EvidenceKind.CPU_CONTRACT, '3b47797fdfc594bb51a349137a0e9faa59ede1fa', '2026-08-26T10:02:52+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, '3b47797fdfc594bb51a349137a0e9faa59ede1fa', '2026-08-26T10:02:52+08:00'),
        ),
    ),
    ('fastvid', 'qwen25_vl', 'forward'): (
        'dc6b26725548a7af1e5bf038afc09df1d83d9e3a17a24500de565c4967a7d69e',
        (
            (EvidenceKind.KERNEL, '35bd55f415624bc2a7aacb3719bb27f74f4604e9', '2026-08-27T22:12:33+08:00'),
            (EvidenceKind.CPU_CONTRACT, '35bd55f415624bc2a7aacb3719bb27f74f4604e9', '2026-08-27T22:12:33+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, '35bd55f415624bc2a7aacb3719bb27f74f4604e9', '2026-08-27T22:12:33+08:00'),
            (EvidenceKind.TRAIN_STEP, '35bd55f415624bc2a7aacb3719bb27f74f4604e9', '2026-08-27T22:12:33+08:00'),
            (EvidenceKind.TRAIN_BACKWARD, '35bd55f415624bc2a7aacb3719bb27f74f4604e9', '2026-08-27T22:12:33+08:00'),
            (EvidenceKind.CHECKPOINT_RELOAD, '35bd55f415624bc2a7aacb3719bb27f74f4604e9', '2026-08-27T22:12:33+08:00'),
        ),
    ),
    ('fastvid', 'qwen25_vl', 'generate'): (
        '0ccf531347620551180af9452e3dada9df86cf444378a94b109fa8a61d6d554f',
        (
            (EvidenceKind.KERNEL, '992ba51355e90d5b86a1d9901e7a2a03135a98a9', '2026-08-26T10:44:00+08:00'),
            (EvidenceKind.CPU_CONTRACT, '992ba51355e90d5b86a1d9901e7a2a03135a98a9', '2026-08-26T10:44:00+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, '992ba51355e90d5b86a1d9901e7a2a03135a98a9', '2026-08-26T10:44:00+08:00'),
        ),
    ),
    ('fastvid', 'qwen3_vl', 'forward'): (
        'b3565277a1463a19f00eb11e2d23cdeb8c81e79aa0f0a1f9dc2bfd4d9b6f1309',
        (
            (EvidenceKind.KERNEL, '5d41a78db8ae5c1edb7dd791227ed60c99f03e37', '2026-08-27T15:00:11+08:00'),
            (EvidenceKind.CPU_CONTRACT, '5d41a78db8ae5c1edb7dd791227ed60c99f03e37', '2026-08-27T15:00:11+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, '017647ea9fe625384ee252ad3a4b30be1556e730', '2026-08-27T15:00:11+08:00'),
            (EvidenceKind.TRAIN_STEP, '017647ea9fe625384ee252ad3a4b30be1556e730', '2026-08-27T15:00:11+08:00'),
            (EvidenceKind.TRAIN_BACKWARD, '017647ea9fe625384ee252ad3a4b30be1556e730', '2026-08-27T15:00:11+08:00'),
            (EvidenceKind.CHECKPOINT_RELOAD, '017647ea9fe625384ee252ad3a4b30be1556e730', '2026-08-27T15:00:11+08:00'),
        ),
    ),
    ('fastvid', 'qwen3_vl', 'generate'): (
        '433a9a5537f0b8e32b597e3f9a93788608acc677508463f4b462b30d367f29f3',
        (
            (EvidenceKind.KERNEL, '5d41a78db8ae5c1edb7dd791227ed60c99f03e37', '2026-08-26T11:26:20+08:00'),
            (EvidenceKind.CPU_CONTRACT, '5d41a78db8ae5c1edb7dd791227ed60c99f03e37', '2026-08-26T11:26:20+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, '5d41a78db8ae5c1edb7dd791227ed60c99f03e37', '2026-08-26T11:26:20+08:00'),
        ),
    ),
    ('vqtoken', 'llava_onevision', 'forward'): (
        'd87b682e51c7765a2da8bd88e1759cffc2e87c50e90e8e8f8af0954dbd23f7f6',
        (
            (EvidenceKind.KERNEL, '99d6c31247ddf9f37cecdce88c542d90b0e381ca', '2026-08-28T00:18:00+08:00'),
            (EvidenceKind.CPU_CONTRACT, '3bcfd8befedc92cd9dd76c4b5f93bbdb7df7ace4', '2026-08-28T00:18:00+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, '9e1890263f3ffd51bda9cfbd1038c5ab001e059d', '2026-08-28T00:18:00+08:00'),
            (EvidenceKind.TRAIN_STEP, '9e1890263f3ffd51bda9cfbd1038c5ab001e059d', '2026-08-28T00:18:00+08:00'),
            (EvidenceKind.TRAIN_BACKWARD, '9e1890263f3ffd51bda9cfbd1038c5ab001e059d', '2026-08-28T00:18:00+08:00'),
            (EvidenceKind.CHECKPOINT_RELOAD, '9e1890263f3ffd51bda9cfbd1038c5ab001e059d', '2026-08-28T00:18:00+08:00'),
        ),
    ),
    ('vqtoken', 'llava_onevision', 'generate'): (
        '7b0306396a0d0c8898ed064c3114f5a4a89eb3eedf78e7d3435bb03cfc200fab',
        (
            (EvidenceKind.KERNEL, '99d6c31247ddf9f37cecdce88c542d90b0e381ca', '2026-08-26T06:06:43+08:00'),
            (EvidenceKind.CPU_CONTRACT, '99d6c31247ddf9f37cecdce88c542d90b0e381ca', '2026-08-26T06:06:43+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, '3bcfd8befedc92cd9dd76c4b5f93bbdb7df7ace4', '2026-08-26T06:06:43+08:00'),
        ),
    ),
    ('vqtoken', 'qwen25_vl', 'forward'): (
        '626dfabec8ec6887b9933a72f31898b7aaf9e3fcc33ad7e37b4581102243c8a2',
        (
            (EvidenceKind.KERNEL, '99d6c31247ddf9f37cecdce88c542d90b0e381ca', '2026-08-27T11:24:00+08:00'),
            (EvidenceKind.CPU_CONTRACT, '6485f0da59684a7facd05d6dcbee5bb60ade057d', '2026-08-27T11:24:00+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, '505027010c4b09584d685f5c1c12fb76fc7bf0b9', '2026-08-27T11:24:00+08:00'),
            (EvidenceKind.TRAIN_STEP, '505027010c4b09584d685f5c1c12fb76fc7bf0b9', '2026-08-27T11:24:00+08:00'),
            (EvidenceKind.TRAIN_BACKWARD, '505027010c4b09584d685f5c1c12fb76fc7bf0b9', '2026-08-27T11:24:00+08:00'),
            (EvidenceKind.CHECKPOINT_RELOAD, '505027010c4b09584d685f5c1c12fb76fc7bf0b9', '2026-08-27T11:24:00+08:00'),
        ),
    ),
    ('vqtoken', 'qwen25_vl', 'generate'): (
        '85b72207be0ea2d41b5792d37006c64c544ba008f7fe556f38c4fe7916ac60c6',
        (
            (EvidenceKind.KERNEL, '99d6c31247ddf9f37cecdce88c542d90b0e381ca', '2026-08-26T06:53:09+08:00'),
            (EvidenceKind.CPU_CONTRACT, '6485f0da59684a7facd05d6dcbee5bb60ade057d', '2026-08-26T06:53:09+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, '6485f0da59684a7facd05d6dcbee5bb60ade057d', '2026-08-26T06:53:09+08:00'),
        ),
    ),
    ('vqtoken', 'qwen3_vl', 'forward'): (
        'c64912fd28d4e3cef580e7052346ea6fa64b24986ce839513cad6febd75503c1',
        (
            (EvidenceKind.KERNEL, '99d6c31247ddf9f37cecdce88c542d90b0e381ca', '2026-08-27T11:53:00+08:00'),
            (EvidenceKind.CPU_CONTRACT, 'ca16fddabe56774b67fce03c18097808317815f0', '2026-08-27T11:53:00+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, 'd1fceb3a0528e448170e6b6287857f38503c1b66', '2026-08-27T11:53:00+08:00'),
            (EvidenceKind.TRAIN_STEP, 'd1fceb3a0528e448170e6b6287857f38503c1b66', '2026-08-27T11:53:00+08:00'),
            (EvidenceKind.TRAIN_BACKWARD, 'd1fceb3a0528e448170e6b6287857f38503c1b66', '2026-08-27T11:53:00+08:00'),
            (EvidenceKind.CHECKPOINT_RELOAD, 'd1fceb3a0528e448170e6b6287857f38503c1b66', '2026-08-27T11:53:00+08:00'),
        ),
    ),
    ('vqtoken', 'qwen3_vl', 'generate'): (
        'ecbbc36ab9eb4b29caca584958c49ac6718b205a26ac298c7c1901345b1fc0ed',
        (
            (EvidenceKind.KERNEL, '99d6c31247ddf9f37cecdce88c542d90b0e381ca', '2026-08-26T07:36:00+08:00'),
            (EvidenceKind.CPU_CONTRACT, 'ca16fddabe56774b67fce03c18097808317815f0', '2026-08-26T07:36:00+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, 'ca16fddabe56774b67fce03c18097808317815f0', '2026-08-26T07:36:00+08:00'),
        ),
    ),
    ('dytok', 'llava_onevision', 'forward'): (
        '4717c6ae50a15ab25d80a5b71d80eabac948f7fcaf54cd57118204bc3620919f',
        (
            (EvidenceKind.KERNEL, 'cd219f64ff22dd53f1d566ac0d1802ccbf6cb0fd', '2026-08-27T20:03:22+08:00'),
            (EvidenceKind.CPU_CONTRACT, 'cd219f64ff22dd53f1d566ac0d1802ccbf6cb0fd', '2026-08-27T20:03:22+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, 'cd219f64ff22dd53f1d566ac0d1802ccbf6cb0fd', '2026-08-27T20:03:22+08:00'),
            (EvidenceKind.TRAIN_STEP, 'cd219f64ff22dd53f1d566ac0d1802ccbf6cb0fd', '2026-08-27T20:03:22+08:00'),
            (EvidenceKind.TRAIN_BACKWARD, 'cd219f64ff22dd53f1d566ac0d1802ccbf6cb0fd', '2026-08-27T20:03:22+08:00'),
            (EvidenceKind.CHECKPOINT_RELOAD, 'cd219f64ff22dd53f1d566ac0d1802ccbf6cb0fd', '2026-08-27T20:03:22+08:00'),
        ),
    ),
    ('dytok', 'llava_onevision', 'generate'): (
        '06e97b0834cfda5256681ff027736ed9c42b28b2361e7442ed72c514cb99961d',
        (
            (EvidenceKind.KERNEL, 'b2a0f76187b8a97e2cb84ba92d60618f01db9c8c', '2026-08-26T18:00:16+08:00'),
            (EvidenceKind.CPU_CONTRACT, 'b2a0f76187b8a97e2cb84ba92d60618f01db9c8c', '2026-08-26T18:00:16+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, 'b2a0f76187b8a97e2cb84ba92d60618f01db9c8c', '2026-08-26T18:00:16+08:00'),
        ),
    ),
    ('dytok', 'qwen25_vl', 'forward'): (
        '558b0a052ecc335c7c349f80935dc33e2eeaa287caa13a16ee633925f13b5051',
        (
            (EvidenceKind.KERNEL, '4cd8c0bea0bf3fadfe4dc84317b07ceabc3eaa28', '2026-08-27T20:40:48+08:00'),
            (EvidenceKind.CPU_CONTRACT, '4cd8c0bea0bf3fadfe4dc84317b07ceabc3eaa28', '2026-08-27T20:40:48+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, '4cd8c0bea0bf3fadfe4dc84317b07ceabc3eaa28', '2026-08-27T20:40:48+08:00'),
            (EvidenceKind.TRAIN_STEP, '4cd8c0bea0bf3fadfe4dc84317b07ceabc3eaa28', '2026-08-27T20:40:48+08:00'),
            (EvidenceKind.TRAIN_BACKWARD, '4cd8c0bea0bf3fadfe4dc84317b07ceabc3eaa28', '2026-08-27T20:40:48+08:00'),
            (EvidenceKind.CHECKPOINT_RELOAD, '4cd8c0bea0bf3fadfe4dc84317b07ceabc3eaa28', '2026-08-27T20:40:48+08:00'),
        ),
    ),
    ('dytok', 'qwen25_vl', 'generate'): (
        '8f4a1a7cec2f67cfe17c29ac4c70bef2dc66a914efc4f9d98b550ca9fddf9d59',
        (
            (EvidenceKind.KERNEL, '1ee315620355f122d2e8305643c6bf6bf8a7aa47', '2026-08-26T20:52:05+08:00'),
            (EvidenceKind.CPU_CONTRACT, '1ee315620355f122d2e8305643c6bf6bf8a7aa47', '2026-08-26T20:52:05+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, '1ee315620355f122d2e8305643c6bf6bf8a7aa47', '2026-08-26T20:52:05+08:00'),
        ),
    ),
    ('dytok', 'qwen3_vl', 'forward'): (
        '81137e0aaf28e0e47722c82c7e7e95bf7bdc5025f04ce966b4836e762a390a48',
        (
            (EvidenceKind.KERNEL, 'cfeba91be57bf251fa4d39eee7c97f0df89a5abe', '2026-08-27T21:25:14+08:00'),
            (EvidenceKind.CPU_CONTRACT, 'cfeba91be57bf251fa4d39eee7c97f0df89a5abe', '2026-08-27T21:25:14+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, 'cfeba91be57bf251fa4d39eee7c97f0df89a5abe', '2026-08-27T21:25:14+08:00'),
            (EvidenceKind.TRAIN_STEP, 'cfeba91be57bf251fa4d39eee7c97f0df89a5abe', '2026-08-27T21:25:14+08:00'),
            (EvidenceKind.TRAIN_BACKWARD, 'cfeba91be57bf251fa4d39eee7c97f0df89a5abe', '2026-08-27T21:25:14+08:00'),
            (EvidenceKind.CHECKPOINT_RELOAD, 'cfeba91be57bf251fa4d39eee7c97f0df89a5abe', '2026-08-27T21:25:14+08:00'),
        ),
    ),
    ('dytok', 'qwen3_vl', 'generate'): (
        '7f5fbd457a9441f27fdf4607e4823ea702caf84497388f6133cedf4e20104fdd',
        (
            (EvidenceKind.KERNEL, 'b7f8f1a753cd34e83f08465935fc963d4372bb77', '2026-08-27T09:24:35+08:00'),
            (EvidenceKind.CPU_CONTRACT, 'b7f8f1a753cd34e83f08465935fc963d4372bb77', '2026-08-27T09:24:35+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, 'b7f8f1a753cd34e83f08465935fc963d4372bb77', '2026-08-27T09:24:35+08:00'),
        ),
    ),
    ('longvu', 'llava_onevision', 'forward'): (
        '41248c8e3ba2bbca8036502c73451e6ff7342b6558b354d5bbad67e51cde3b50',
        (
            (EvidenceKind.KERNEL, '28f379863fa66cbd71cf22bc47db7a92ba164201', '2026-09-06T20:27:38Z'),
            (EvidenceKind.CPU_CONTRACT, '28f379863fa66cbd71cf22bc47db7a92ba164201', '2026-09-06T20:27:38Z'),
            (EvidenceKind.REAL_CHECKPOINT, '28f379863fa66cbd71cf22bc47db7a92ba164201', '2026-09-06T20:27:38Z'),
            (EvidenceKind.TRAIN_STEP, '28f379863fa66cbd71cf22bc47db7a92ba164201', '2026-09-06T20:27:38Z'),
            (EvidenceKind.TRAIN_BACKWARD, '28f379863fa66cbd71cf22bc47db7a92ba164201', '2026-09-06T20:27:38Z'),
            (EvidenceKind.CHECKPOINT_RELOAD, '28f379863fa66cbd71cf22bc47db7a92ba164201', '2026-09-06T20:27:38Z'),
        ),
    ),
    ('longvu', 'llava_onevision', 'generate'): (
        '759cebc3a756f4d58a9bd2d6ccdc2d1529c2d2bef635ccafbce6e11be0f393e6',
        (
            (EvidenceKind.KERNEL, '9a362e6d186a68820a9e746e402a3e27ca449562', '2026-08-28T16:13:29+08:00'),
            (EvidenceKind.CPU_CONTRACT, '9a362e6d186a68820a9e746e402a3e27ca449562', '2026-08-28T16:13:29+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, '9a362e6d186a68820a9e746e402a3e27ca449562', '2026-08-28T16:13:29+08:00'),
        ),
    ),
    ('longvu', 'qwen25_vl', 'forward'): (
        'bee3e176c7892e3cddd6859214b848e863c5458e72a03046123131bf3ab8daf9',
        (
            (EvidenceKind.KERNEL, '28f379863fa66cbd71cf22bc47db7a92ba164201', '2026-09-06T20:29:29Z'),
            (EvidenceKind.CPU_CONTRACT, '28f379863fa66cbd71cf22bc47db7a92ba164201', '2026-09-06T20:29:29Z'),
            (EvidenceKind.REAL_CHECKPOINT, '28f379863fa66cbd71cf22bc47db7a92ba164201', '2026-09-06T20:29:29Z'),
            (EvidenceKind.TRAIN_STEP, '28f379863fa66cbd71cf22bc47db7a92ba164201', '2026-09-06T20:29:29Z'),
            (EvidenceKind.TRAIN_BACKWARD, '28f379863fa66cbd71cf22bc47db7a92ba164201', '2026-09-06T20:29:29Z'),
            (EvidenceKind.CHECKPOINT_RELOAD, '28f379863fa66cbd71cf22bc47db7a92ba164201', '2026-09-06T20:29:29Z'),
        ),
    ),
    ('longvu', 'qwen25_vl', 'generate'): (
        '0bfc2fabb7695f4d058cd27ed45552d691839bccefb927a39717eecfc6fda8b4',
        (
            (EvidenceKind.KERNEL, 'ced0dc5731566c4aa1e1c178f3bbf792d3202874', '2026-08-28T16:43:28+08:00'),
            (EvidenceKind.CPU_CONTRACT, 'ced0dc5731566c4aa1e1c178f3bbf792d3202874', '2026-08-28T16:43:28+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, 'ced0dc5731566c4aa1e1c178f3bbf792d3202874', '2026-08-28T16:43:28+08:00'),
        ),
    ),
    ('longvu', 'qwen3_vl', 'forward'): (
        '6cf97a6cabdf4390d73e2cfda2170dd875a0a95ffd2a3fa6678861c2b0057408',
        (
            (EvidenceKind.KERNEL, '28f379863fa66cbd71cf22bc47db7a92ba164201', '2026-09-06T20:32:29Z'),
            (EvidenceKind.CPU_CONTRACT, '28f379863fa66cbd71cf22bc47db7a92ba164201', '2026-09-06T20:32:29Z'),
            (EvidenceKind.REAL_CHECKPOINT, '28f379863fa66cbd71cf22bc47db7a92ba164201', '2026-09-06T20:32:29Z'),
            (EvidenceKind.TRAIN_STEP, '28f379863fa66cbd71cf22bc47db7a92ba164201', '2026-09-06T20:32:29Z'),
            (EvidenceKind.TRAIN_BACKWARD, '28f379863fa66cbd71cf22bc47db7a92ba164201', '2026-09-06T20:32:29Z'),
            (EvidenceKind.CHECKPOINT_RELOAD, '28f379863fa66cbd71cf22bc47db7a92ba164201', '2026-09-06T20:32:29Z'),
        ),
    ),
    ('longvu', 'qwen3_vl', 'generate'): (
        'e5984207a157e0d8b3c40c7eaf38b1640e26cd9fa0ac8e4cd584773752f00ad3',
        (
            (EvidenceKind.KERNEL, 'e47b9118e1a0decd403f1b50ef4cf3121ab30f49', '2026-08-28T17:13:12+08:00'),
            (EvidenceKind.CPU_CONTRACT, 'e47b9118e1a0decd403f1b50ef4cf3121ab30f49', '2026-08-28T17:13:12+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, 'e47b9118e1a0decd403f1b50ef4cf3121ab30f49', '2026-08-28T17:13:12+08:00'),
        ),
    ),
    ('llamavid', 'llava_onevision', 'forward'): (
        'fae5c6b46a8ae336aaa5a55e0f8b84d309ca3424ced4743a1db9a68e4ff68e69',
        (
            (EvidenceKind.KERNEL, '28f379863fa66cbd71cf22bc47db7a92ba164201', '2026-09-06T20:27:19Z'),
            (EvidenceKind.CPU_CONTRACT, '28f379863fa66cbd71cf22bc47db7a92ba164201', '2026-09-06T20:27:19Z'),
            (EvidenceKind.REAL_CHECKPOINT, '28f379863fa66cbd71cf22bc47db7a92ba164201', '2026-09-06T20:27:19Z'),
            (EvidenceKind.TRAIN_STEP, '28f379863fa66cbd71cf22bc47db7a92ba164201', '2026-09-06T20:27:19Z'),
            (EvidenceKind.TRAIN_BACKWARD, '28f379863fa66cbd71cf22bc47db7a92ba164201', '2026-09-06T20:27:19Z'),
            (EvidenceKind.CHECKPOINT_RELOAD, '28f379863fa66cbd71cf22bc47db7a92ba164201', '2026-09-06T20:27:19Z'),
        ),
    ),
    ('llamavid', 'llava_onevision', 'generate'): (
        '62847c049a82db4260e7a652e6e6220ecd689f4e088ea63bfc6ba52587d84e20',
        (
            (EvidenceKind.KERNEL, 'eb941dbcdcb56a8b3d751c605761e905d17e5cf1', '2026-08-28T19:40:00+08:00'),
            (EvidenceKind.CPU_CONTRACT, 'eb941dbcdcb56a8b3d751c605761e905d17e5cf1', '2026-08-28T19:40:00+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, 'eb941dbcdcb56a8b3d751c605761e905d17e5cf1', '2026-08-28T19:40:00+08:00'),
        ),
    ),
    ('llamavid', 'qwen25_vl', 'forward'): (
        '65d121926fb156c2a2a3f117d4206dfec7453f883f1d7e3a7a43f262aed469db',
        (
            (EvidenceKind.KERNEL, '28f379863fa66cbd71cf22bc47db7a92ba164201', '2026-09-06T20:29:27Z'),
            (EvidenceKind.CPU_CONTRACT, '28f379863fa66cbd71cf22bc47db7a92ba164201', '2026-09-06T20:29:27Z'),
            (EvidenceKind.REAL_CHECKPOINT, '28f379863fa66cbd71cf22bc47db7a92ba164201', '2026-09-06T20:29:27Z'),
            (EvidenceKind.TRAIN_STEP, '28f379863fa66cbd71cf22bc47db7a92ba164201', '2026-09-06T20:29:27Z'),
            (EvidenceKind.TRAIN_BACKWARD, '28f379863fa66cbd71cf22bc47db7a92ba164201', '2026-09-06T20:29:27Z'),
            (EvidenceKind.CHECKPOINT_RELOAD, '28f379863fa66cbd71cf22bc47db7a92ba164201', '2026-09-06T20:29:27Z'),
        ),
    ),
    ('llamavid', 'qwen25_vl', 'generate'): (
        'e53d7f20372c0ac73e08cb615c372156bb75cf70fadec56525476c4a32b752c7',
        (
            (EvidenceKind.KERNEL, '6d7efa194f94e173ff6e4146c3e86ee4ba5929df', '2026-09-06T20:17:11Z'),
            (EvidenceKind.CPU_CONTRACT, '6d7efa194f94e173ff6e4146c3e86ee4ba5929df', '2026-09-06T20:17:11Z'),
            (EvidenceKind.REAL_CHECKPOINT, '6d7efa194f94e173ff6e4146c3e86ee4ba5929df', '2026-09-06T20:17:11Z'),
        ),
    ),
    ('llamavid', 'qwen3_vl', 'forward'): (
        '5d86acb102a41b0f73448f523f18c429dc0606d35dbd505abf09918d6745e01a',
        (
            (EvidenceKind.KERNEL, '28f379863fa66cbd71cf22bc47db7a92ba164201', '2026-09-06T20:32:24Z'),
            (EvidenceKind.CPU_CONTRACT, '28f379863fa66cbd71cf22bc47db7a92ba164201', '2026-09-06T20:32:24Z'),
            (EvidenceKind.REAL_CHECKPOINT, '28f379863fa66cbd71cf22bc47db7a92ba164201', '2026-09-06T20:32:24Z'),
            (EvidenceKind.TRAIN_STEP, '28f379863fa66cbd71cf22bc47db7a92ba164201', '2026-09-06T20:32:24Z'),
            (EvidenceKind.TRAIN_BACKWARD, '28f379863fa66cbd71cf22bc47db7a92ba164201', '2026-09-06T20:32:24Z'),
            (EvidenceKind.CHECKPOINT_RELOAD, '28f379863fa66cbd71cf22bc47db7a92ba164201', '2026-09-06T20:32:24Z'),
        ),
    ),
    ('llamavid', 'qwen3_vl', 'generate'): (
        '16526031206592bf09e0753043914ba25c881bb2f0d3010a064ee3e529f684f6',
        (
            (EvidenceKind.KERNEL, '6d7efa194f94e173ff6e4146c3e86ee4ba5929df', '2026-09-06T20:17:01Z'),
            (EvidenceKind.CPU_CONTRACT, '6d7efa194f94e173ff6e4146c3e86ee4ba5929df', '2026-09-06T20:17:01Z'),
            (EvidenceKind.REAL_CHECKPOINT, '6d7efa194f94e173ff6e4146c3e86ee4ba5929df', '2026-09-06T20:17:01Z'),
        ),
    ),
    ('dycoke', 'llava_onevision', 'forward'): (
        '510c86821848f9ad38a24be1d47009a298dffcb90f711c68e8ae954a3cbd1842',
        (
            (EvidenceKind.KERNEL, 'f656b67350407a6df98648074ba5e9e8262b138d', '2026-08-28T15:28:41+08:00'),
            (EvidenceKind.CPU_CONTRACT, 'f656b67350407a6df98648074ba5e9e8262b138d', '2026-08-28T15:28:41+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, 'f656b67350407a6df98648074ba5e9e8262b138d', '2026-08-28T15:28:41+08:00'),
            (EvidenceKind.TRAIN_STEP, 'f656b67350407a6df98648074ba5e9e8262b138d', '2026-08-28T15:28:41+08:00'),
            (EvidenceKind.TRAIN_BACKWARD, 'f656b67350407a6df98648074ba5e9e8262b138d', '2026-08-28T15:28:41+08:00'),
            (EvidenceKind.CHECKPOINT_RELOAD, 'f656b67350407a6df98648074ba5e9e8262b138d', '2026-08-28T15:28:41+08:00'),
        ),
    ),
    ('dycoke', 'llava_onevision', 'generate'): (
        '54bad0695ecff05934127c6b5926b2de2abeab40eaf0091a26722f690645fad6',
        (
            (EvidenceKind.KERNEL, '19199a7ea65f278efc3eabfd1d84c0b754ef158e', '2026-08-27T09:50:34+08:00'),
            (EvidenceKind.CPU_CONTRACT, '19199a7ea65f278efc3eabfd1d84c0b754ef158e', '2026-08-27T09:50:34+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, '19199a7ea65f278efc3eabfd1d84c0b754ef158e', '2026-08-27T09:50:34+08:00'),
        ),
    ),
    ('dycoke', 'qwen25_vl', 'forward'): (
        '35c5da705c9386ad02d212602271b61d3157616bb7dff353265df38c35ebbdc3',
        (
            (EvidenceKind.KERNEL, 'f656b67350407a6df98648074ba5e9e8262b138d', '2026-08-28T15:28:41+08:00'),
            (EvidenceKind.CPU_CONTRACT, 'f656b67350407a6df98648074ba5e9e8262b138d', '2026-08-28T15:28:41+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, 'f656b67350407a6df98648074ba5e9e8262b138d', '2026-08-28T15:28:41+08:00'),
            (EvidenceKind.TRAIN_STEP, 'f656b67350407a6df98648074ba5e9e8262b138d', '2026-08-28T15:28:41+08:00'),
            (EvidenceKind.TRAIN_BACKWARD, 'f656b67350407a6df98648074ba5e9e8262b138d', '2026-08-28T15:28:41+08:00'),
            (EvidenceKind.CHECKPOINT_RELOAD, 'f656b67350407a6df98648074ba5e9e8262b138d', '2026-08-28T15:28:41+08:00'),
        ),
    ),
    ('dycoke', 'qwen25_vl', 'generate'): (
        'ee4a6f33345d41129042e09608f35d258b2b2afc673f426e8f4dd162b0fa29da',
        (
            (EvidenceKind.KERNEL, '2e170dda7a1c9eeb51f594f42627ba4f960f3ab6', '2026-08-27T10:13:51+08:00'),
            (EvidenceKind.CPU_CONTRACT, '2e170dda7a1c9eeb51f594f42627ba4f960f3ab6', '2026-08-27T10:13:51+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, '2e170dda7a1c9eeb51f594f42627ba4f960f3ab6', '2026-08-27T10:13:51+08:00'),
        ),
    ),
    ('dycoke', 'qwen3_vl', 'forward'): (
        'ec96a3f2baf2852d9dfd2cd60c9d619c5e5a77b38a1e8fa28df68d58bf491166',
        (
            (EvidenceKind.KERNEL, 'f656b67350407a6df98648074ba5e9e8262b138d', '2026-08-28T15:28:41+08:00'),
            (EvidenceKind.CPU_CONTRACT, 'f656b67350407a6df98648074ba5e9e8262b138d', '2026-08-28T15:28:41+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, 'f656b67350407a6df98648074ba5e9e8262b138d', '2026-08-28T15:28:41+08:00'),
            (EvidenceKind.TRAIN_STEP, 'f656b67350407a6df98648074ba5e9e8262b138d', '2026-08-28T15:28:41+08:00'),
            (EvidenceKind.TRAIN_BACKWARD, 'f656b67350407a6df98648074ba5e9e8262b138d', '2026-08-28T15:28:41+08:00'),
            (EvidenceKind.CHECKPOINT_RELOAD, 'f656b67350407a6df98648074ba5e9e8262b138d', '2026-08-28T15:28:41+08:00'),
        ),
    ),
    ('dycoke', 'qwen3_vl', 'generate'): (
        'ff32242eb98dc11c4fee641f5eef4020288913d201400bc8970a79a68eeaf393',
        (
            (EvidenceKind.KERNEL, '2ec9be4cdb1bddaa1e6f22fa65ca5a359db91e3f', '2026-08-27T10:39:23+08:00'),
            (EvidenceKind.CPU_CONTRACT, '2ec9be4cdb1bddaa1e6f22fa65ca5a359db91e3f', '2026-08-27T10:39:23+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, '2ec9be4cdb1bddaa1e6f22fa65ca5a359db91e3f', '2026-08-27T10:39:23+08:00'),
        ),
    ),
    ('vqtoken', 'qwen3_5', 'generate'): (
        'b83c144b015efd08caa12d7320da27385ad30356ddc152b28f712f38ba68523c',
        (
            (EvidenceKind.KERNEL, '7e3ae876bc6306a4179f389e5f960225cd9e3a79', '2026-10-12T11:37:12+08:00'),
            (EvidenceKind.CPU_CONTRACT, '7e3ae876bc6306a4179f389e5f960225cd9e3a79', '2026-10-12T11:37:12+08:00'),
            (EvidenceKind.REAL_CHECKPOINT, '7e3ae876bc6306a4179f389e5f960225cd9e3a79', '2026-10-12T11:37:12+08:00'),
        ),
    ),
})


def video_evidence(algorithm: str, backbone: str, phase: str) -> tuple[EvidenceRecord, ...]:
    """Project only explicitly recorded provenance into support gates."""

    reference = VIDEO_EVIDENCE.get((algorithm, backbone, phase))
    if reference is None:
        return ()
    checksum, records = reference
    artifact = f"docs/support/video_compression/evidence/{algorithm}_{backbone}_{phase}.json"
    return tuple(
        EvidenceRecord(
            kind=kind, passed=True, artifact=artifact, checksum=checksum,
            source_commit=revision, recorded_at=recorded_at,
        )
        for kind, revision, recorded_at in records
    )
