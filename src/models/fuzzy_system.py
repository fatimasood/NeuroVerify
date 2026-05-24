"""
Fuzzy logic-based decision making system as per paper Section 3.5
Implements rules from Table 2 and Equation (5), (6), (7), (8)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FuzzyMembership(nn.Module):
    """
    Fuzzy membership functions as per paper Section 3.5
    Gaussian membership functions from Equation (6)
    """
    
    def __init__(self, num_inputs=2, num_mfs=3):
        super(FuzzyMembership, self).__init__()
        self.num_inputs = num_inputs
        self.num_mfs = num_mfs
        
        # Gaussian membership parameters (center, sigma)
        # Paper Equation (6): μ(x) = exp(-(x-c)²/(2σ²))
        self.centers = nn.Parameter(torch.tensor([[0.2, 0.5, 0.8],
                                                   [0.2, 0.5, 0.8]]))
        self.sigmas = nn.Parameter(torch.tensor([[0.15, 0.2, 0.15],
                                                  [0.15, 0.2, 0.15]]))
    
    def gaussian_mf(self, x, center, sigma):
        """Gaussian membership function as per paper Equation (6)"""
        return torch.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    
    def forward(self, x):
        """
        Compute membership degrees
        Args:
            x: (batch_size, num_inputs)
        Returns:
            membership: (batch_size, num_inputs, num_mfs)
        """
        batch_size = x.shape[0]
        memberships = []
        
        for i in range(self.num_inputs):
            input_mfs = []
            for j in range(self.num_mfs):
                mf = self.gaussian_mf(x[:, i], self.centers[i, j], self.sigmas[i, j])
                input_mfs.append(mf)
            memberships.append(torch.stack(input_mfs, dim=1))
        
        return torch.stack(memberships, dim=1)


class FuzzyRuleLayer(nn.Module):
    """
    Fuzzy rule evaluation as per paper Section 3.5
    Implements rules from Table 2
    """
    
    def __init__(self, num_inputs=2, num_mfs=3, num_rules=4):
        super(FuzzyRuleLayer, self).__init__()
        self.num_inputs = num_inputs
        self.num_mfs = num_mfs
        self.num_rules = num_rules
        
        # Rule antecedents (based on Table 2)
        # 0=Low(Fake), 1=Medium, 2=High(Real)
        # Rule 1: IF text IS real (High) AND image IS real (High) THEN class IS real
        # Rule 2: IF text IS fake (Low) AND image IS real (High) THEN class IS real
        # Rule 3: IF text IS real (High) AND image IS fake (Low) THEN class IS fake
        # Rule 4: IF text IS fake (Low) AND image IS fake (Low) THEN class IS fake
        self.register_buffer(
            'rule_antecedents',
            torch.tensor([[2, 2],   # Rule 1: Text High, Image High -> Real
                         [0, 2],    # Rule 2: Text Low, Image High -> Real
                         [2, 0],    # Rule 3: Text High, Image Low -> Fake
                         [0, 0]])   # Rule 4: Text Low, Image Low -> Fake
        )
        
        # Rule consequents (Real class probability)
        # As per paper Table 2
        self.rule_consequents = nn.Parameter(torch.tensor([
            [0.9, 0.1],  # Rule 1: Strong Real
            [0.7, 0.3],  # Rule 2: Likely Real (image overrides)
            [0.2, 0.8],  # Rule 3: Likely Fake
            [0.1, 0.9]   # Rule 4: Strong Fake
        ]))
        
        # Rule weights (learnable)
        self.rule_weights = nn.Parameter(torch.ones(num_rules))
    
    def forward(self, memberships):
        """
        Evaluate fuzzy rules as per paper Equation (5)
        Args:
            memberships: (batch_size, num_inputs, num_mfs)
        Returns:
            rule_firings: (batch_size, num_rules)
            outputs: (batch_size, 2)
        """
        batch_size = memberships.shape[0]
        
        # Compute rule firing strengths (using min for AND as per paper)
        rule_firings = torch.ones(batch_size, self.num_rules, device=memberships.device)
        
        for r in range(self.num_rules):
            for i in range(self.num_inputs):
                mf_idx = self.rule_antecedents[r, i]
                rule_firings[:, r] = torch.minimum(
                    rule_firings[:, r], 
                    memberships[:, i, mf_idx]
                )
        
        # Apply rule weights
        rule_weights = torch.sigmoid(self.rule_weights)
        rule_firings = rule_firings * rule_weights
        
        # Compute weighted outputs (Paper Equation 5)
        # D_f = Σ(μ_k · z_k) / Σ(μ_k)
        total_weight = rule_firings.sum(dim=1, keepdim=True) + 1e-8
        outputs = torch.zeros(batch_size, 2, device=memberships.device)
        
        for r in range(self.num_rules):
            outputs += rule_firings[:, r:r+1] * self.rule_consequents[r]
        
        outputs = outputs / total_weight
        
        return rule_firings, outputs


class FuzzyDecisionSystem(nn.Module):
    """
    Fuzzy logic-based decision making system as per paper Section 3.5
    Implements Figure 3 and Figure 4 logic
    """
    
    def __init__(self, num_rules=4):
        super(FuzzyDecisionSystem, self).__init__()
        
        self.membership = FuzzyMembership(num_inputs=2, num_mfs=3)
        self.rule_layer = FuzzyRuleLayer(num_inputs=2, num_mfs=3, num_rules=num_rules)
        
        # Fuzzy confidence score threshold
        self.threshold = nn.Parameter(torch.tensor(0.5), requires_grad=False)
    
    def forward(self, text_confidence, image_confidence):
        """
        Fuzzy decision making as per paper Section 3.5
        
        Args:
            text_confidence: (batch_size,) confidence score for text (0-1)
            image_confidence: (batch_size,) confidence score for image (0-1)
        Returns:
            prediction: (batch_size,) predicted class (0=fake, 1=real)
            confidence: (batch_size,) final confidence score
            fuzzy_confidence: (batch_size,) fuzzy confidence score (Paper Eq 8)
            rule_firings: (batch_size, num_rules) rule activation strengths
        """
        # Stack inputs
        x = torch.stack([text_confidence, image_confidence], dim=1)
        
        # Fuzzification (convert crisp inputs to fuzzy membership degrees)
        memberships = self.membership(x)
        
        # Rule evaluation and inference
        rule_firings, outputs = self.rule_layer(memberships)
        
        # Defuzzification (centroid method as per paper)
        probs = F.softmax(outputs, dim=1)
        prediction = probs.argmax(dim=1)
        confidence = probs[:, 1]
        
        # Fuzzy confidence score (Paper Equation 8)
        # FCS = Σ(w_i × μ_i) / Σ(w_i)
        rule_weights = torch.sigmoid(self.rule_layer.rule_weights)
        fuzzy_confidence = (rule_firings * rule_weights).sum(dim=1) / (rule_firings.sum(dim=1) + 1e-8)
        
        return prediction, confidence, fuzzy_confidence, rule_firings
    
    def explain_decision(self, text_confidence, image_confidence):
        """
        Provide interpretable explanation of the decision
        As described in paper Section 3.5 and Figure 4
        """
        with torch.no_grad():
            prediction, confidence, fuzzy_conf, rule_firings = self.forward(
                torch.tensor(text_confidence).view(1, -1) if isinstance(text_confidence, float) else text_confidence,
                torch.tensor(image_confidence).view(1, -1) if isinstance(image_confidence, float) else image_confidence
            )
        
        explanations = []
        batch_size = len(text_confidence) if isinstance(text_confidence, (list, torch.Tensor)) else 1
        
        for i in range(batch_size):
            pred = prediction[i].item() if hasattr(prediction, '__getitem__') else prediction.item()
            conf = confidence[i].item() if hasattr(confidence, '__getitem__') else confidence.item()
            fuzzy = fuzzy_conf[i].item() if hasattr(fuzzy_conf, '__getitem__') else fuzzy_conf.item()
            text_c = text_confidence[i].item() if isinstance(text_confidence, (list, torch.Tensor)) else text_confidence
            img_c = image_confidence[i].item() if isinstance(image_confidence, (list, torch.Tensor)) else image_confidence
            
            # Find most activated rule
            rule_firing = rule_firings[i] if hasattr(rule_firings, '__getitem__') else rule_firings
            max_rule = rule_firing.argmax().item()
            
            rule_explanations = {
                0: "✓ Text is REAL AND Image is REAL → Decision: REAL (Both modalities agree)",
                1: "⚠ Text is FAKE BUT Image is REAL → Decision: REAL (Image evidence overrides text)",
                2: "⚠ Text is REAL BUT Image is FAKE → Decision: FAKE (Image evidence overrides text)",
                3: "✗ Text is FAKE AND Image is FAKE → Decision: FAKE (Both modalities agree)"
            }
            
            explanation = {
                'decision': 'REAL' if pred == 1 else 'FAKE',
                'confidence': float(conf),
                'fuzzy_confidence': float(fuzzy),
                'text_confidence': float(text_c),
                'image_confidence': float(img_c),
                'activated_rule': rule_explanations[max_rule],
                'rule_firing_strength': float(rule_firing[max_rule])
            }
            explanations.append(explanation)
        
        return explanations if batch_size > 1 else explanations[0]
    
    def get_decision_path(self, text_confidence, image_confidence):
        """
        Get the decision path as shown in Figure 4 of the paper
        """
        explanation = self.explain_decision(text_confidence, image_confidence)
        
        if isinstance(explanation, dict):
            print("\n" + "="*60)
            print("FUZZY DECISION PATH (As per Figure 4 in paper)")
            print("="*60)
            print(f"Input: Text Confidence = {explanation['text_confidence']:.3f}, Image Confidence = {explanation['image_confidence']:.3f}")
            print(f"\nStep 1 - Fuzzification:")
            print(f"  → Text membership: {'High (Real)' if explanation['text_confidence'] > 0.6 else 'Medium' if explanation['text_confidence'] > 0.4 else 'Low (Fake)'}")
            print(f"  → Image membership: {'High (Real)' if explanation['image_confidence'] > 0.6 else 'Medium' if explanation['image_confidence'] > 0.4 else 'Low (Fake)'}")
            print(f"\nStep 2 - Rule Evaluation:")
            print(f"  → {explanation['activated_rule']}")
            print(f"\nStep 3 - Defuzzification:")
            print(f"  → Final Confidence: {explanation['confidence']:.3f}")
            print(f"  → Final Decision: {explanation['decision']}")
            print("="*60)
        
        return explanation