import torch

from models.mlp import MLP


class DETR(torch.nn.Module):
    """
    DETR Model from: <https://arxiv.org/pdf/2005.12872.pdf>
    Code adapted from: <https://github.com/facebookresearch/detr/blob/master/models/detr.py>
    """
    def __init__(self, backbone, transformer, hidden_dim=256, num_classes=41, num_queries=25):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            positional_embedding: torch module of the positional embedding to be used. See positional_embedding.py
            transformer: torch module of the transformer architecture.
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image.
        """
        super().__init__()
        self.backbone = backbone

        self.backbone_projection = torch.nn.Conv2d(2048, hidden_dim, kernel_size=1)

        self.positional_embedding = torch.nn.Embedding(1200, hidden_dim)
        self.positional_indices = torch.arange(0, 1200)

        self.num_queries = num_queries
        self.query_indices = torch.arange(0, num_queries)
        self.query_embedding = torch.nn.Embedding(num_queries, hidden_dim)
        self.transformer = transformer

        self.class_embed = torch.nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, images: torch.Tensor):
        """The forward expects:
               - images: batched images, of shape [batch_size x 3 x H x W]

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
        """
        backbone_output = self.backbone(images)
        backbone_projection = self.backbone_projection(backbone_output).flatten(2)

        positional_embedding = self.positional_embedding(self.positional_indices)
        positional_embedding = positional_embedding.repeat(images.shape[0], 1, 1).permute(0, 2, 1)

        transformer_input = backbone_projection + positional_embedding
        # Transformer input is provided as [batch, sequence, feature]
        transformer_input = transformer_input.permute(0, 2, 1)

        query_embedding = self.query_embedding(self.query_indices).repeat(images.shape[0], 1, 1)

        transformer_output = self.transformer(src=transformer_input, tgt=query_embedding)

        outputs_class = self.class_embed(transformer_output).sigmoid()
        outputs_bbox = self.bbox_embed(transformer_output).sigmoid()

        out = {'class_probabilities': outputs_class, 'boxes': outputs_bbox}

        return out
